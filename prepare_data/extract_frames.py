"""
Extract frames from all videos in a folder and save as PNG images.

Usage:
    python prepare_data/extract_frames.py --input_dir /path/to/videos --output_dir /path/to/frames --interval 10

Args:
    --input_dir:   Directory containing video files.
    --output_dir:  Directory to save extracted PNG frames.
    --interval:    Frame interval. Extract one frame every N frames. Default: 1 (every frame).
    --by_time:     If set, interpret --interval as seconds instead of frame count.
"""

import argparse
import os
import glob
import cv2


VIDEO_EXTENSIONS = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.flv", "*.wmv", "*.webm", "*.m4v")


def get_video_files(input_dir: str) -> list[str]:
    files = []
    for ext in VIDEO_EXTENSIONS:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
        files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    return sorted(set(files))


def extract_frames(
    video_path: str,
    output_dir: str,
    interval: int = 1,
    by_time: bool = False,
) -> int:
    """Extract frames from a single video.

    Args:
        video_path:  Path to the video file.
        output_dir:  Directory to save frames.
        interval:    Frame interval (frames) or time interval (seconds) depending on by_time.
        by_time:     If True, interval is in seconds; otherwise in frame count.

    Returns:
        Number of frames saved.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = os.path.join(output_dir, video_name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARNING] Cannot open video: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if by_time:
        frame_step = max(1, int(round(fps * interval)))
    else:
        frame_step = max(1, int(interval))

    print(f"  Video: {video_path}")
    print(f"    FPS: {fps:.2f}, Total frames: {total_frames}, Step: {frame_step}")

    saved = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step == 0:
            filename = os.path.join(save_dir, f"{frame_idx:06d}.png")
            cv2.imwrite(filename, frame)
            saved += 1
        frame_idx += 1

    cap.release()
    print(f"    Saved {saved} frames -> {save_dir}")
    return saved


def extract_all(
    input_dir: str,
    output_dir: str,
    interval: int = 1,
    by_time: bool = False,
):
    """Extract frames from all videos in a directory."""
    video_files = get_video_files(input_dir)
    if not video_files:
        print(f"No video files found in {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Found {len(video_files)} video(s) in {input_dir}")

    total_saved = 0
    for vf in video_files:
        total_saved += extract_frames(vf, output_dir, interval, by_time)

    print(f"\nDone. Total frames saved: {total_saved}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from videos as PNG images.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing video files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save extracted frames.")
    parser.add_argument("--interval", type=float, default=1,
                        help="Frame interval. Default 1 = every frame. With --by_time, unit is seconds.")
    parser.add_argument("--by_time", action="store_true",
                        help="Interpret --interval as seconds instead of frame count.")
    args = parser.parse_args()

    extract_all(args.input_dir, args.output_dir, args.interval, args.by_time)
