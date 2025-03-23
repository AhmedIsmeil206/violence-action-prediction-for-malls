from ultralytics import YOLO

class YOLOv8Detector:
    def __init__(self, model_path):
        # Load your fine-tuned YOLOv8 model
        self.model = YOLO(model_path)

    def detect(self, frame):
        # Run YOLOv8 detection on the frame
        results = self.model(frame)

        # Format detections for DeepSORT
        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes in [x1, y1, x2, y2] format
            confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
            class_ids = result.boxes.cls.cpu().numpy()  # Class IDs

            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                detections.append([box[0], box[1], box[2], box[3], conf, cls_id])

        return detections