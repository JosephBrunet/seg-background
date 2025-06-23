import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
from skimage.measure import block_reduce
from scipy.ndimage import median_filter, label, binary_opening
from sam2.build_sam import build_sam2_video_predictor

import dask
from alive_progress import alive_bar

def fill_2d_holes(volume):
    result = np.zeros_like(volume, dtype=np.uint8)
    print("Filling holes slice-by-slice...")
    for i in tqdm(range(volume.shape[0]), desc="2D Hole Filling"):
        slice_ = volume[i].astype(bool)
        slice_ = binary_opening(slice_, structure=np.ones((20, 20)))
        inverse = ~slice_
        labeled, _ = label(inverse)
        border_labels = np.unique(np.concatenate([
            labeled[0, :].ravel(), labeled[-1, :].ravel(),
            labeled[:, 0].ravel(), labeled[:, -1].ravel()
        ]))
        background_mask = np.isin(labeled, border_labels)
        filled = np.where(background_mask, 0, 1).astype(np.uint8)
        result[i] = np.maximum(slice_,filled)
    return result

def max_upsample(volume, factor, original_shape):
    upsampled = np.repeat(volume, factor, axis=0)
    upsampled = np.repeat(upsampled, factor, axis=1)
    upsampled = np.repeat(upsampled, factor, axis=2)
    return upsampled[:original_shape[0], :original_shape[1], :original_shape[2]]



def main():
    # ========= PARSE ARGUMENTS =========
    parser = argparse.ArgumentParser(description="SAM2 segmentation with mask smoothing.")
    parser.add_argument("im_dir", type=str, help="Path to directory with TIFF or JP2 slices")
    parser.add_argument("--output", type=str, default=".", help="Path to output segmentation directory")
    parser.add_argument("--index", type=int, default=None, help="Index of starting slice")
    parser.add_argument("--window", type=int, nargs=2, default=(1, 99), metavar=("LOW", "HIGH"),
                        help="Percentile range for histogram adjustment, e.g., 1 99")
    parser.add_argument("--downsample", type=int, default=None, help="Optional 3D downsampling factor (e.g. 2)")
    parser.add_argument("--model", type=str, choices=["tiny", "small", "base", "large"], default="small",
                        help="SAM2 model variant (default: small)")
    parser.add_argument("--fill-holes", action="store_true",
                        help="Fill holes in the final mask volume before upsampling")
    args = parser.parse_args()

    # ========= CONFIGURATION =========
    im_dir = Path(args.im_dir)
    seg_dir = Path(args.output) / "segmentation_SAM2"
    mask_output_dir = seg_dir / "masks"
    video_forward_path = seg_dir / "video_forward.mp4"
    index = args.index
    window = tuple(args.window)

    # Model selection
    model_suffix = {"tiny": "t", "small": "s", "base": "b", "large": "l"}[args.model]
    project_root = Path(__file__).parents[2].resolve()
    checkpoint_path = project_root / "checkpoints" / f"sam2.1_hiera_{args.model}.pt"
    config_rel_path = Path("configs") / "sam2.1" / f"sam2.1_hiera_{model_suffix}.yaml"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found at {checkpoint_path!r}.")

    fps = 10
    os.makedirs(mask_output_dir, exist_ok=True)

    # ========== LOAD AND NORMALIZE STACK ==========
    print("Loading image stack...")
    im_files = sorted([f for f in im_dir.iterdir() if f.suffix.lower() in [".tif", ".jp2"]])
    if len(im_files) == 0:
        print(f"Error: No .tif/.jp2 files found in {im_dir}")
        return

    stack = []
    def read_slice(path):
        img = cv2.imread(str(path), -1)
        if img is None:
            print(f"Warning: Could not read {path.name}; skipping.")
            raise( ValueError(f"Could not read image {path.name}"))
        return img

    with alive_bar(len(im_files), title="Reading slices", length=40) as bar:
        delayed_images = [
            dask.delayed(lambda f=path: (bar(), read_slice(f))[1])()
            for path in im_files
        ]
        da_stack = dask.compute(*delayed_images)
        
    stack = np.stack(da_stack, axis=0)


    if len(stack) == 0:
        print("Error: No valid images loaded. Exiting.")
        return

    stack_np = np.stack(stack, axis=0)
    original_shape = stack_np.shape

    if args.downsample:
        print(f"Applying 3D downsampling by factor {args.downsample}...")
        block_size = (args.downsample,) * 3
        stack_np = block_reduce(stack_np, block_size=block_size, func=np.mean).astype(np.float32)
    else:
        print("Applying 2D median filtering to each slice...")
        stack_np = np.stack(
            [median_filter(img, size=2) for img in tqdm(stack_np, desc="2D Filtering")], axis=0
        )

    if not (0 <= window[0] < window[1] <= 100):
        parser.error(f"Invalid --window range: {window}. Must satisfy 0 <= low < high <= 100.")
    vmin = np.percentile(stack_np, window[0]).astype(np.float32)
    vmax = np.percentile(stack_np, window[1]).astype(np.float32)
    stack_normalized = np.clip((stack_np - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)

    print("Applying 3D median filter after normalization...")
    stack_normalized = median_filter(stack_normalized, size=(3, 3, 3))

    height, width = stack_normalized.shape[1:]
    start_idx = len(stack_normalized) // 2 if index is None else index

    # ========= WRITE VIDEOS ==========
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    print("Writing forward video...")
    writer = cv2.VideoWriter(str(video_forward_path), fourcc, fps, (width, height), isColor=True)
    for frame in stack_normalized:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB))
    writer.release()

    # ========= LOAD SAM2 ==========
    print(f"Loading SAM2 model from:\n  checkpoint: {checkpoint_path}\n  config:     {config_rel_path}")
    predictor = build_sam2_video_predictor(str(config_rel_path), str(checkpoint_path))

    print(f"Segmenting frame #{start_idx}...")
    frame_rgb = cv2.cvtColor(stack_normalized[start_idx], cv2.COLOR_GRAY2RGB)
    roi = cv2.selectROI("Select ROI on middle slice", frame_rgb, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    x, y, w, h = roi
    box = [x, y, x + w, y + h]

    # ========= FORWARD PROPAGATION ==========
    print("Propagating forward...")
    state = predictor.init_state(str(video_forward_path))
    _, obj_ids, _ = predictor.add_new_points_or_box(state, frame_idx=start_idx, box=np.array(box), obj_id=1)

    forward_masks = []
    for _, _, mask_logits in tqdm(predictor.propagate_in_video(state), desc="Forward Propagation"):
        for mask_tensor in mask_logits:
            mask_np = (mask_tensor > 0).cpu().numpy().astype(np.uint8)
            forward_masks.append(np.squeeze(mask_np))

    # ========= COMBINE MASKS ==========
    print("Combining and filling mask stack...")

    mask_stack_3d = np.stack(forward_masks, axis=0)
    if args.fill_holes:
        print("Filling internal holes in 2D mask slices...")
        mask_stack_3d = fill_2d_holes(mask_stack_3d)
        
    if args.downsample:
        print("Upsampling mask volume back to original resolution using max-style upsampling...")
        mask_stack_3d = max_upsample(mask_stack_3d, args.downsample, original_shape)

    print("Smoothing and writing masks...")
    kernel = np.ones((3, 3), np.uint8)
    for i in range(mask_stack_3d.shape[0]):
        neighbors = [mask_stack_3d[i]]
        if i > 0:
            neighbors.append(mask_stack_3d[i - 1])
        if i < mask_stack_3d.shape[0] - 1:
            neighbors.append(mask_stack_3d[i + 1])
        avg = np.mean(neighbors, axis=0)
        smoothed = (avg > 0.5).astype(np.uint8)
        dilated = cv2.dilate(smoothed, kernel, iterations=1) * 255
        out_path = mask_output_dir / f"mask_{i:04d}.png"
        cv2.imwrite(str(out_path), dilated)

    print("All masks saved to:", mask_output_dir)


if __name__ == "__main__":
    main()
