"""VisDrone object detection module for OmniUAV.

Provides object detection capabilities using VisDrone-trained models.
Supports local video files and real-time frame processing.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    fasterrcnn_mobilenet_v3_large_fpn,
    fcos_resnet50_fpn,
    retinanet_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.fcos import FCOSClassificationHead

# VisDrone class names
VISDRONE_CLASSES = [
    "ignored-regions", "pedestrian", "people", "bicycle",
    "car", "van", "truck", "tricycle", "awning-tricycle",
    "bus", "motor", "others"
]


def get_model(model_name: str, num_classes: int, pretrained: bool = True):
    """Get detection model by name.

    Args:
        model_name: Model architecture name
        num_classes: Number of classes (including background)
        pretrained: Whether to use pretrained weights

    Returns:
        Detection model
    """
    # Use weights parameter instead of deprecated pretrained
    weights = "DEFAULT" if pretrained else None

    if model_name == "fasterrcnn_resnet50":
        model = fasterrcnn_resnet50_fpn(weights=weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    elif model_name == "fasterrcnn_mobilenet":
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    elif model_name == "fcos_resnet50":
        model = fcos_resnet50_fpn(weights=weights)
        in_channels = model.head.classification_head.conv[0].in_channels
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = FCOSClassificationHead(
            in_channels, num_anchors, num_classes
        )

    elif model_name == "retinanet_resnet50":
        model = retinanet_resnet50_fpn(weights=weights)
        in_channels = model.head.classification_head.conv[0][0].in_channels
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels, num_anchors, num_classes
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


class VisDroneDetector:
    """VisDrone object detector for video frames."""

    def __init__(
        self,
        model_name: str = "fasterrcnn_resnet50",
        checkpoint_path: Optional[str] = None,
        num_classes: int = 12,
        device: Optional[str] = None,
        score_threshold: float = 0.5,
    ):
        """Initialize VisDrone detector.

        Args:
            model_name: Model architecture (fasterrcnn_resnet50, fasterrcnn_mobilenet,
                       fcos_resnet50, retinanet_resnet50)
            checkpoint_path: Path to trained model checkpoint (optional)
            num_classes: Number of classes (default: 12 for VisDrone)
            device: Device to run on (cuda/cpu, auto-detect if None)
            score_threshold: Confidence threshold for detections
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.score_threshold = score_threshold
        self.model_name = model_name
        self.num_classes = num_classes

        print(f"Initializing VisDrone detector on {self.device}...")
        self.model = self._load_model(checkpoint_path, model_name, num_classes)
        print("✓ Model loaded successfully")

    def _load_model(
        self, checkpoint_path: Optional[str], model_name: str, num_classes: int
    ) -> torch.nn.Module:
        """Load detection model."""
        if checkpoint_path:
            print(f"Loading checkpoint from {checkpoint_path}...")
            model = get_model(
                model_name=model_name,
                num_classes=num_classes,
                pretrained=False,
            )
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
        else:
            print("Using pretrained COCO weights...")
            model = get_model(
                model_name=model_name,
                num_classes=num_classes,
                pretrained=True,
            )
            print("Note: Using COCO pretrained weights. Train on VisDrone for better results!")

        model.to(self.device)
        model.eval()
        return model

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run detection on a single frame.

        Args:
            frame: Input frame (BGR format from OpenCV)

        Returns:
            Tuple of (boxes, labels, scores) as numpy arrays
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to tensor
        image_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.to(self.device)

        # Run inference
        with torch.no_grad():
            predictions = self.model([image_tensor])[0]

        # Get predictions
        boxes = predictions["boxes"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()

        # Filter by score threshold
        mask = scores >= self.score_threshold
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]

        return boxes, labels, scores

    def draw_detections(
        self,
        frame: np.ndarray,
        boxes: np.ndarray,
        labels: np.ndarray,
        scores: np.ndarray,
    ) -> np.ndarray:
        """Draw bounding boxes and labels on frame.

        Args:
            frame: Input frame (BGR format)
            boxes: Detection boxes (N, 4) as [x1, y1, x2, y2]
            labels: Class labels (N,)
            scores: Confidence scores (N,)

        Returns:
            Frame with drawn detections
        """
        h, w = frame.shape[:2]
        frame = frame.copy()

        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box.astype(int)

            # Clip to frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Get class name
            class_name = (
                VISDRONE_CLASSES[label]
                if label < len(VISDRONE_CLASSES)
                else f"class_{label}"
            )

            # Choose color based on class
            color = (0, 255, 0)  # Default green
            if label == 1 or label == 2:  # pedestrian, people
                color = (0, 0, 255)  # Red
            elif 4 <= label <= 10:  # vehicles
                color = (255, 0, 0)  # Blue

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label_text = f"{class_name}: {score:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            # Ensure label is within frame
            label_y1 = max(y1 - text_height - 4, 0)
            label_y2 = label_y1 + text_height + 4

            cv2.rectangle(frame, (x1, label_y1), (x1 + text_width, label_y2), color, -1)
            cv2.putText(
                frame,
                label_text,
                (x1, label_y2 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return frame

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, int]:
        """Process a single frame with detection and drawing.

        Args:
            frame: Input frame (BGR format)

        Returns:
            Tuple of (processed frame, number of detections)
        """
        boxes, labels, scores = self.detect(frame)
        frame_with_detections = self.draw_detections(frame, boxes, labels, scores)
        return frame_with_detections, len(boxes)


def process_video(
    video_path: str,
    output_path: Optional[str] = None,
    model_name: str = "fasterrcnn_resnet50",
    checkpoint_path: Optional[str] = None,
    score_threshold: float = 0.5,
    display: bool = True,
    save_video: bool = True,
) -> None:
    """Process a video file with VisDrone detection.

    Args:
        video_path: Path to input video file
        output_path: Path to save output video (optional)
        model_name: Model architecture to use
        checkpoint_path: Path to trained model checkpoint (optional)
        score_threshold: Confidence threshold for detections
        display: Whether to display video during processing
        save_video: Whether to save output video
    """
    # Initialize detector
    detector = VisDroneDetector(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        score_threshold=score_threshold,
    )

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nProcessing video: {video_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")

    # Setup output video writer
    writer = None
    if save_video:
        if output_path is None:
            video_path_obj = Path(video_path)
            output_path = str(
                video_path_obj.parent / f"{video_path_obj.stem}_detected.mp4"
            )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Output will be saved to: {output_path}")

    frame_count = 0
    total_detections = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Process frame
            processed_frame, num_detections = detector.process_frame(frame)
            total_detections += num_detections

            # Add frame info
            info_text = f"Frame: {frame_count}/{total_frames} | Detections: {num_detections}"
            cv2.putText(
                processed_frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # Save frame
            if writer:
                writer.write(processed_frame)

            # Display frame
            if display:
                cv2.imshow("VisDrone Detection", processed_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\nProcessing interrupted by user")
                    break

            # Progress update
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")

    finally:
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

        print(f"\n{'=' * 60}")
        print("Processing Summary:")
        print(f"  Frames processed: {frame_count}")
        print(f"  Total detections: {total_detections}")
        print(f"  Average detections per frame: {total_detections / frame_count:.2f}")
        if save_video and output_path:
            print(f"  Output saved to: {output_path}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VisDrone video detection")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--output", help="Path to output video (optional)")
    parser.add_argument(
        "--model",
        default="fasterrcnn_resnet50",
        choices=[
            "fasterrcnn_resnet50",
            "fasterrcnn_mobilenet",
            "fcos_resnet50",
            "retinanet_resnet50",
        ],
        help="Model architecture",
    )
    parser.add_argument("--checkpoint", help="Path to model checkpoint (optional)")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display video during processing",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save output video",
    )

    args = parser.parse_args()

    process_video(
        video_path=args.video,
        output_path=args.output,
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        score_threshold=args.score_threshold,
        display=not args.no_display,
        save_video=not args.no_save,
    )
