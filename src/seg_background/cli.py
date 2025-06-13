import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
from scipy.ndimage import median_filter
from sam2.build_sam import build_sam2_video_predictor


def main():
    # ========= PARSE ARGUMENTS =========
    parser = argparse.ArgumentParser(description="SAM2 segmentation with mask smoothing.")
    parser.add_argument(
        "im_dir",
        type=str,
        help="Path to directory with TIFF or JP2 slices"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=".",
        help="Path to output segmentation directory"
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Index of starting slice"
    )
    parser.add_argument(
        "--window",
        type=int,
        nargs=2,
        default=(1, 99),
        help="Percentile range for histogram adjustment as two integers (0,100)"
    )
    args = parser.parse_args()

    # ========= CONFIGURATION =========
    im_dir = Path(args.im_dir)
    seg_dir = Path(args.output) / "segmentation_SAM2"
    mask_output_dir = seg_dir / "masks"
    video_forward_path = seg_dir / "video_forward.mp4"
    video_backward_path = seg_dir / "video_backward.mp4"
    index = args.index
    window = tuple(args.window)

    # ─── Determine project root (two levels up from this file)
    project_root = Path(__file__).parents[2].resolve()

    # ─── Absolute paths to checkpoint and config under project root
    checkpoint_path = project_root / "checkpoints" / "sam2.1_hiera_large.pt"
    config_rel_path = Path("configs") / "sam2.1" / "sam2.1_hiera_l.yaml"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"SAM2 checkpoint not found at {checkpoint_path!r}.\n"
            "Please place sam2.1_hiera_large.pt in the top-level checkpoints/ folder."
        )


    fps = 10
    os.makedirs(mask_output_dir, exist_ok=True)

    # ========== LOAD AND NORMALIZE STACK ==========
    print("Loading image stack...")
    im_files = sorted(
        [f for f in im_dir.iterdir() if f.suffix.lower() in [".tif", ".jp2"]]
    )
    if len(im_files) == 0:
        print(f"Error: No .tif/.jp2 files found in {im_dir}")
        return

    stack = []
    for f in tqdm(im_files, desc="Reading slices"):
        img = cv2.imread(str(f), -1)
        if img is None:
            print(f"Warning: Could not read {f.name}; skipping.")
            continue
        stack.append(img)

    if len(stack) == 0:
        print("Error: No valid images loaded. Exiting.")
        return

    print("Applying median filter to each slice...")
    stack_filtered = []
    for img in tqdm(stack, desc="Median filtering slices"):
        stack_filtered.append(median_filter(img, size=2))

    stack_np = np.stack(stack_filtered, axis=0)

    if not (0 <= window[0] < window[1] <= 100):
        parser.error(f"Invalid --window range: {window}. Must satisfy 0 <= low < high <= 100.")

    vmin = np.percentile(stack_np, window[0])
    vmax = np.percentile(stack_np, window[1])
    stack_normalized = np.clip(
        (stack_np - vmin) / (vmax - vmin) * 255, 0, 255
    ).astype(np.uint8)

    print("Applying 3D median filter...")
    stack_normalized = median_filter(stack_normalized, size=(3, 3, 3))

    height, width = stack_normalized.shape[1:]

    # Define starting point 
    if index == None: 
        start_idx = len(stack_normalized) // 2
    else:   
        start_idx = index

    # ========== WRITE VIDEOS ==========
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    print("Writing forward video...")
    os.makedirs(seg_dir, exist_ok=True)
    writer = cv2.VideoWriter(
        str(video_forward_path), fourcc, fps, (width, height), isColor=True
    )
    for frame in stack_normalized[start_idx:]:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()

    print("Writing backward video...")
    writer = cv2.VideoWriter(
        str(video_backward_path), fourcc, fps, (width, height), isColor=True
    )
    for frame in stack_normalized[start_idx - 1 :: -1]:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()

    # ========== LOAD SAM2 ==========
    # Change working directory so Hydra can locate configs/ relative to project root

    print(
        f"Loading SAM2 model from:\n"
        f"  checkpoint: {checkpoint_path}\n"
        f"  config:     {config_rel_path}"
    )
    predictor = build_sam2_video_predictor(
        str(config_rel_path),
        str(checkpoint_path)
    )

    # ========== MIDDLE SLICE SELECTION ==========
    print("Segmenting middle frame...")
    frame_bgr = cv2.cvtColor(
        stack_normalized[start_idx], cv2.COLOR_GRAY2BGR
    )
    roi = cv2.selectROI(
        "Select ROI on middle slice", frame_bgr,
        fromCenter=False, showCrosshair=True
    )
    cv2.destroyAllWindows()
    x, y, w, h = roi
    box = [x, y, x + w, y + h]

    # ========== FORWARD PROPAGATION ==========
    print("Propagating forward...")
    state = predictor.init_state(str(video_forward_path))
    _, obj_ids, mask_logits = predictor.add_new_points_or_box(
        state, frame_idx=0, box=np.array(box), obj_id=1
    )

    forward_masks = []
    for _, _, mask_logits in tqdm(
        predictor.propagate_in_video(state),
        desc="Forward Propagation"
    ):
        for mask_tensor in mask_logits:
            mask_np = (mask_tensor > 0).cpu().numpy().astype(np.uint8)
            mask_np = np.squeeze(mask_np)
            if mask_np.shape != (height, width):
                mask_np = cv2.resize(
                    mask_np, (width, height),
                    interpolation=cv2.INTER_NEAREST
                )
            forward_masks.append(mask_np)

    # ========== BACKWARD PROPAGATION ==========
    print("Propagating backward...")
    state = predictor.init_state(str(video_backward_path))
    _, obj_ids, mask_logits = predictor.add_new_points_or_box(
        state, frame_idx=0, box=np.array(box), obj_id=1
    )

    backward_masks = []
    for _, _, mask_logits in tqdm(
        predictor.propagate_in_video(state),
        desc="Backward Propagation"
    ):
        for mask_tensor in mask_logits:
            mask_np = (mask_tensor > 0).cpu().numpy().astype(np.uint8)
            mask_np = np.squeeze(mask_np)
            if mask_np.shape != (height, width):
                mask_np = cv2.resize(
                    mask_np, (width, height),
                    interpolation=cv2.INTER_NEAREST
                )
            backward_masks.append(mask_np)

    # ========== COMBINE MASKS ==========
    print("Smoothing and writing masks with neighbor averaging and dilation...")

    mask_stack = [None] * len(stack_normalized)
    for i, mask in enumerate(forward_masks):
        idx = start_idx + i
        if 0 <= idx < len(mask_stack):
            mask_stack[idx] = mask
    for i, mask in enumerate(backward_masks):
        idx = start_idx - 1 - i
        if 0 <= idx < len(mask_stack):
            mask_stack[idx] = mask

    # Fill missing frames with zero‐mask
    mask_stack = [
        (m if m is not None else np.zeros((height, width), dtype=np.uint8))
        for m in mask_stack
    ]

    kernel = np.ones((3, 3), np.uint8)
    for i in range(len(mask_stack)):
        neighbors = [mask_stack[i]]
        if i > 0:
            neighbors.append(mask_stack[i - 1])
        if i < len(mask_stack) - 1:
            neighbors.append(mask_stack[i + 1])
        avg = np.mean(neighbors, axis=0)
        smoothed = (avg > 0.5).astype(np.uint8)
        dilated = cv2.dilate(smoothed, kernel, iterations=1) * 255
        out_path = mask_output_dir / f"mask_{i:04d}.png"
        cv2.imwrite(str(out_path), dilated)

    print("✅ All masks saved to:", mask_output_dir)


if __name__ == "__main__":
    main()
