"""
Build real temporal landmark sequences from sign-language videos.

Expected input layout:
    DATASET/
        CLASS_A/
            video1.mp4
            video2.mp4
        CLASS_B/
            video3.mp4

Each video produces one or more samples shaped (timesteps, 63).
No frame is repeated to fake temporal information.

Example:
    python src/build_sequence_dataset.py ^
        --input D:\Datasets\asl_subset ^
        --output data\temporal\asl_subset.npz ^
        --timesteps 23 ^
        --frame-stride 2 ^
        --max-videos-per-class 100
"""

import argparse
import os
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def extract_landmark_sequence(video_path, hands, frame_stride=1):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open: {video_path}")
        return None

    sequence = []
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % frame_stride != 0:
            frame_index += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            coords = []
            for lm in hand.landmark:
                coords.extend((lm.x, lm.y, lm.z))

            sequence.append(np.asarray(coords, dtype=np.float32))

        frame_index += 1

    cap.release()

    if not sequence:
        return None

    return np.asarray(sequence, dtype=np.float32)


def make_windows(sequence, timesteps, windows_per_video=1):
    if len(sequence) < timesteps:
        return []

    max_start = len(sequence) - timesteps

    if windows_per_video <= 1 or max_start == 0:
        starts = [0 if max_start == 0 else max_start // 2]
    else:
        count = min(windows_per_video, max_start + 1)
        starts = np.linspace(0, max_start, count, dtype=int)
        starts = np.unique(starts).tolist()

    return [sequence[start:start + timesteps] for start in starts]


def normalize_sequence(sequence):
    """Match the current project's inference normalization."""
    sequence = sequence.astype(np.float32, copy=True)
    max_value = np.max(np.abs(sequence))
    if max_value > 0:
        sequence /= max_value
    return sequence


def discover_videos(root, max_videos_per_class=None, selected_classes=None):
    root = Path(root)
    class_dirs = sorted(p for p in root.iterdir() if p.is_dir())

    if selected_classes:
        selected = set(selected_classes)
        class_dirs = [p for p in class_dirs if p.name in selected]

    videos = []
    class_names = [p.name for p in class_dirs]

    for class_index, class_dir in enumerate(class_dirs):
        class_videos = sorted(
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )

        if max_videos_per_class is not None:
            class_videos = class_videos[:max_videos_per_class]

        for video in class_videos:
            videos.append((video, class_index, class_dir.name))

    return videos, class_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Root directory containing one folder per class.")
    parser.add_argument("--output", required=True, help="Output .npz file.")
    parser.add_argument("--timesteps", type=int, default=23)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--windows-per-video", type=int, default=1)
    parser.add_argument("--max-videos-per-class", type=int, default=None)
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Optional class names to process. If omitted, all class folders are used.",
    )
    args = parser.parse_args()

    if args.timesteps < 2:
        raise ValueError("timesteps must be >= 2")
    if args.frame_stride < 1:
        raise ValueError("frame-stride must be >= 1")

    videos, class_names = discover_videos(
        args.input,
        args.max_videos_per_class,
        args.classes,
    )

    if not videos:
        raise RuntimeError(
            "No videos found. Expected input/<class>/*.mp4 (or another supported video extension)."
        )

    print(f"Classes: {class_names}")
    print(f"Videos found: {len(videos)}")
    print(f"Target sequence: ({args.timesteps}, 63)")

    X = []
    y = []
    groups = []
    skipped = 0

    mp_hands = mp.solutions.hands

    # static_image_mode=False keeps MediaPipe in video/tracking mode.
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:

        for index, (video_path, class_index, class_name) in enumerate(videos, start=1):
            print(f"[{index}/{len(videos)}] {class_name}: {video_path.name}")

            raw_sequence = extract_landmark_sequence(
                video_path,
                hands,
                frame_stride=args.frame_stride,
            )

            if raw_sequence is None:
                print("    -> skipped: no hand landmarks detected")
                skipped += 1
                continue

            windows = make_windows(
                raw_sequence,
                args.timesteps,
                args.windows_per_video,
            )

            if not windows:
                print(
                    f"    -> skipped: only {len(raw_sequence)} detected frames; "
                    f"need at least {args.timesteps}"
                )
                skipped += 1
                continue

            relative_group = str(video_path.relative_to(Path(args.input)))

            for window in windows:
                X.append(normalize_sequence(window))
                y.append(class_index)
                groups.append(relative_group)

            print(
                f"    -> detected frames: {len(raw_sequence)}, "
                f"sequences created: {len(windows)}"
            )

    if not X:
        raise RuntimeError("No valid temporal sequences were created.")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output,
        X=X,
        y=y,
        groups=groups,
        class_names=np.asarray(class_names),
        timesteps=np.asarray(args.timesteps),
        features=np.asarray(63),
    )

    print("\nDone.")
    print(f"Saved: {output}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Unique videos/groups: {len(np.unique(groups))}")
    print(f"Skipped videos: {skipped}")


if __name__ == "__main__":
    main()
