# yolov8_detector.py
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from deep_sort.detection import Detection  # Add this import

class YOLOv8Detector:
    def __init__(self, model_path, reid_model=None):
        self.model = YOLO(model_path)
        self.reid_model = reid_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.reid_model:
            self.reid_model = self.reid_model.to(self.device)
            self.reid_model.eval()
            self.transform = transforms.Compose([
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

    def detect(self, image):
        """Detect objects in image and return list of Detection objects"""
        results = self.model(image)
        detections = []
        
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()

            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                if cls_id != 0:  # Only keep person detections
                    continue
                    
                bbox = [box[0], box[1], box[2]-box[0], box[3]-box[1]]
                feature = np.array([])
                
                if self.reid_model:
                    # Extract ReID features
                    img_patch = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    if img_patch.size > 0:
                        try:
                            img_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB)
                            img_patch = Image.fromarray(img_patch)
                            img_tensor = self.transform(img_patch).unsqueeze(0).to(self.device)
                            with torch.no_grad():
                                feature = self.reid_model(img_tensor, return_features=True)
                                feature = feature.cpu().numpy().flatten()
                        except Exception as e:
                            print(f"Error extracting ReID features: {e}")
                
                detections.append(Detection(bbox, float(conf), feature))
        
        return detections