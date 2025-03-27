import os
import cv2
import argparse
import numpy as np
from yolov8.yolov8_detector import YOLOv8Detector
from deep_sort import nn_matching
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection  # Ensure this import exists
from market1501_reid import load_reid_model

def generate_videos(mot_dir, output_dir, model_path, fps=30):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load ReID model
        reid_model = load_reid_model()
        print("ReID model loaded successfully")
    except Exception as e:
        print(f"Error loading ReID model: {e}")
        print("Falling back to tracking without ReID features")
        reid_model = None

    # Initialize YOLOv8 detector with your fine-tuned model
    detector = YOLOv8Detector(model_path, reid_model)

    # Initialize DeepSORT tracker
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", 0.3, None)  # metric_type, matching_threshold, budget
    tracker = Tracker(metric, max_iou_distance=0.7, max_age=30, n_init=3)

    # Iterate over all sequences in the MOT directory
    for sequence_name in os.listdir(mot_dir):
        sequence_dir = os.path.join(mot_dir, sequence_name)
        image_dir = os.path.join(sequence_dir, "img1")

        # Check if the sequence directory exists
        if not os.path.exists(sequence_dir):
            print(f"Error: Sequence directory not found: {sequence_dir}")
            continue

        # Read image paths
        image_paths = {
            int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
            for f in sorted(os.listdir(image_dir))
            if f.endswith(('.jpg', '.png', '.jpeg'))
        }
        if not image_paths:
            print(f"Error: No images found in {image_dir}")
            continue

        # Create video writer
        first_image = cv2.imread(next(iter(image_paths.values())))
        if first_image is None:
            print(f"Error: Unable to read first image in {image_dir}")
            continue

        height, width, _ = first_image.shape
        output_video_path = os.path.join(output_dir, f"{sequence_name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for .mp4 files
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Get total number of frames
        total_frames = len(image_paths)
        print(f"Generating video for sequence: {sequence_name}")
        print(f"Total frames to process: {total_frames}")

        # Generate video
        for frame_idx, frame_num in enumerate(sorted(image_paths.keys()), start=1):
            # Read image
            image = cv2.imread(image_paths[frame_num])
            if image is None:
                print(f"Error: Unable to read image {image_paths[frame_num]}")
                continue

            # Generate detections using YOLOv8
            detections = detector.detect(image)

            # Format detections for DeepSORT
            deep_sort_detections = []
            for det in detections:
                if isinstance(det, Detection):  # Already processed by detector with ReID
                    deep_sort_detections.append(det)
                else:  # Fallback for non-ReID detections
                    try:
                        x1, y1, x2, y2, confidence, class_id = det
                        if class_id != 0:  # Filter out non-human detections
                            continue
                        bbox = [x1, y1, x2 - x1, y2 - y1]
                        deep_sort_detections.append(Detection(bbox, confidence, np.array([])))
                    except Exception as e:
                        print(f"Error processing detection: {e}")
                        continue

            # Update DeepSORT tracker
            tracker.predict()
            tracker.update(deep_sort_detections)

            # Draw bounding boxes and IDs
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlwh()  # Get bounding box in [x, y, width, height] format
                x1, y1, w, h = map(int, bbox)
                track_id = track.track_id  # Get unique track ID
                cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
                cv2.putText(image, f"ID: {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Write frame to video
            out.write(image)

            # Print progress
            print(f"Processed frame {frame_idx}/{total_frames} (Frame ID: {frame_num})", end="\r")

        # Release video writer
        out.release()
        print(f"\nVideo saved: {output_video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mot_dir", help="Path to MOTChallenge directory", required=True)
    parser.add_argument("--output_dir", help="Path to output video directory", required=True)
    parser.add_argument("--model_path", help="Path to fine-tuned YOLOv8 model", required=True)
    parser.add_argument("--fps", help="Frame rate (FPS) for the output video", type=int, default=30)
    args = parser.parse_args()

    # Convert model_path to absolute path
    model_path = os.path.abspath(args.model_path)
    generate_videos(args.mot_dir, args.output_dir, model_path, args.fps)