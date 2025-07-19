"""
YOLO-based object detection for poker game elements.
Provides real-time detection of cards, chips, players, and markers.
"""

import cv2
import numpy as np
import torch
import time
from typing import List, Tuple, Dict, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import json
import requests
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PokerObjectType(Enum):
    """Types of poker objects that can be detected"""
    CARD = "card"
    CHIP = "chip"
    PLAYER = "player"
    DEALER_BUTTON = "dealer_button"
    BETTING_MARKER = "betting_marker"
    ACTION_BUTTON = "action_button"

@dataclass
class YOLODetection:
    """YOLO detection result for poker objects"""
    object_type: PokerObjectType
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    center: Tuple[int, int]
    class_id: int
    label: str
    
    # Additional poker-specific attributes
    card_rank: Optional[str] = None
    card_suit: Optional[str] = None
    chip_value: Optional[int] = None
    player_id: Optional[int] = None

@dataclass
class PokerGameDetection:
    """Complete poker game state from YOLO detection"""
    cards: List[YOLODetection]
    chips: List[YOLODetection]
    players: List[YOLODetection]
    markers: List[YOLODetection]
    action_buttons: List[YOLODetection]
    detection_time: float
    frame_size: Tuple[int, int]

class YOLOPokerDetector:
    """
    YOLO-based poker game object detector.
    Provides real-time detection and classification of poker game elements.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 config_path: Optional[str] = None,
                 weights_path: Optional[str] = None,
                 use_gpu: bool = True):
        """
        Initialize YOLO detector for poker objects.
        
        Args:
            model_path: Path to YOLO model file
            config_path: Path to YOLO config file
            weights_path: Path to pre-trained weights
            use_gpu: Whether to use GPU acceleration
        """
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # YOLO model configuration
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        self.input_size = (640, 640)  # YOLO input size
        
        # Initialize model
        self._initialize_model(model_path, config_path, weights_path)
        
        # Poker-specific class mapping
        self.poker_classes = self._load_poker_classes()
        
        # Performance metrics
        self.detection_times = []
        self.frame_count = 0
        
    def _initialize_model(self, model_path: str, config_path: str, weights_path: str):
        """Initialize YOLO model"""
        try:
            # Try to load YOLOv5 (PyTorch-based)
            if model_path and os.path.exists(model_path):
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
                logger.info(f"Loaded custom YOLO model from {model_path}")
            else:
                # Use pre-trained YOLOv5 and adapt for poker objects
                self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                logger.info("Loaded pre-trained YOLOv5s model")
            
            self.model.to(self.device)
            self.model.eval()
            
            # Configure model for poker detection
            self.model.conf = self.confidence_threshold
            self.model.iou = self.nms_threshold
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            # Fallback to OpenCV DNN if available
            self._initialize_opencv_dnn(config_path, weights_path)
    
    def _initialize_opencv_dnn(self, config_path: str, weights_path: str):
        """Initialize OpenCV DNN backend as fallback"""
        try:
            if config_path and weights_path and os.path.exists(config_path) and os.path.exists(weights_path):
                self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                logger.info("Initialized OpenCV DNN YOLO backend")
                self.use_opencv_dnn = True
            else:
                logger.warning("No valid YOLO model found, using mock detector")
                self.use_mock = True
                self.use_opencv_dnn = False
        except Exception as e:
            logger.error(f"Failed to initialize OpenCV DNN: {e}")
            self.use_mock = True
            self.use_opencv_dnn = False
    
    def _load_poker_classes(self) -> Dict[str, int]:
        """Load poker-specific class mappings"""
        # Define poker object classes
        poker_classes = {
            # Cards (52 classes for each card + face-down)
            'card_2h': 0, 'card_3h': 1, 'card_4h': 2, 'card_5h': 3, 'card_6h': 4,
            'card_7h': 5, 'card_8h': 6, 'card_9h': 7, 'card_th': 8, 'card_jh': 9,
            'card_qh': 10, 'card_kh': 11, 'card_ah': 12,
            
            'card_2d': 13, 'card_3d': 14, 'card_4d': 15, 'card_5d': 16, 'card_6d': 17,
            'card_7d': 18, 'card_8d': 19, 'card_9d': 20, 'card_td': 21, 'card_jd': 22,
            'card_qd': 23, 'card_kd': 24, 'card_ad': 25,
            
            'card_2c': 26, 'card_3c': 27, 'card_4c': 28, 'card_5c': 29, 'card_6c': 30,
            'card_7c': 31, 'card_8c': 32, 'card_9c': 33, 'card_tc': 34, 'card_jc': 35,
            'card_qc': 36, 'card_kc': 37, 'card_ac': 38,
            
            'card_2s': 39, 'card_3s': 40, 'card_4s': 41, 'card_5s': 42, 'card_6s': 43,
            'card_7s': 44, 'card_8s': 45, 'card_9s': 46, 'card_ts': 47, 'card_js': 48,
            'card_qs': 49, 'card_ks': 50, 'card_as': 51,
            
            'card_back': 52,  # Face-down card
            
            # Chips (different denominations)
            'chip_1': 53, 'chip_5': 54, 'chip_10': 55, 'chip_25': 56, 'chip_50': 57,
            'chip_100': 58, 'chip_500': 59, 'chip_1000': 60, 'chip_5000': 61,
            
            # Players and markers
            'player': 62, 'dealer_button': 63, 'small_blind': 64, 'big_blind': 65,
            
            # Action buttons
            'fold_button': 66, 'call_button': 67, 'raise_button': 68, 'check_button': 69,
            'bet_button': 70, 'all_in_button': 71,
            
            # Table elements
            'pot': 72, 'board_area': 73, 'player_area': 74
        }
        
        return poker_classes
    
    def detect_poker_objects(self, image: np.ndarray) -> PokerGameDetection:
        """
        Detect all poker objects in the given image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            PokerGameDetection with all detected objects
        """
        start_time = time.time()
        
        if hasattr(self, 'use_mock') and self.use_mock:
            # Mock detection for testing
            detections = self._mock_detection(image)
        elif hasattr(self, 'use_opencv_dnn') and self.use_opencv_dnn:
            # OpenCV DNN detection
            detections = self._detect_opencv_dnn(image)
        else:
            # PyTorch YOLO detection
            detections = self._detect_pytorch_yolo(image)
        
        detection_time = time.time() - start_time
        self.detection_times.append(detection_time)
        self.frame_count += 1
        
        # Categorize detections
        categorized = self._categorize_detections(detections)
        
        return PokerGameDetection(
            cards=categorized['cards'],
            chips=categorized['chips'],
            players=categorized['players'],
            markers=categorized['markers'],
            action_buttons=categorized['action_buttons'],
            detection_time=detection_time,
            frame_size=(image.shape[1], image.shape[0])
        )
    
    def _detect_pytorch_yolo(self, image: np.ndarray) -> List[YOLODetection]:
        """Detect objects using PyTorch YOLO model"""
        detections = []
        
        try:
            # Preprocess image
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run inference
            results = self.model(rgb_image)
            
            # Parse results
            for *box, conf, cls in results.xyxy[0].cpu().numpy():
                if conf > self.confidence_threshold:
                    x1, y1, x2, y2 = map(int, box)
                    width = x2 - x1
                    height = y2 - y1
                    center_x = x1 + width // 2
                    center_y = y1 + height // 2
                    
                    # Map class to poker object
                    class_id = int(cls)
                    label = self._get_class_label(class_id)
                    object_type = self._get_object_type(label)
                    
                    detection = YOLODetection(
                        object_type=object_type,
                        confidence=float(conf),
                        bbox=(x1, y1, width, height),
                        center=(center_x, center_y),
                        class_id=class_id,
                        label=label
                    )
                    
                    # Add poker-specific attributes
                    self._extract_poker_attributes(detection)
                    detections.append(detection)
                    
        except Exception as e:
            logger.error(f"PyTorch YOLO detection failed: {e}")
            
        return detections
    
    def _detect_opencv_dnn(self, image: np.ndarray) -> List[YOLODetection]:
        """Detect objects using OpenCV DNN backend"""
        detections = []
        
        try:
            # Create blob from image
            blob = cv2.dnn.blobFromImage(image, 1/255.0, self.input_size, swapRB=True, crop=False)
            self.net.setInput(blob)
            
            # Run inference
            layer_outputs = self.net.forward(self._get_output_layers())
            
            # Parse outputs
            boxes = []
            confidences = []
            class_ids = []
            
            for output in layer_outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > self.confidence_threshold:
                        # Scale bbox to image size
                        h, w = image.shape[:2]
                        center_x = int(detection[0] * w)
                        center_y = int(detection[1] * h)
                        width = int(detection[2] * w)
                        height = int(detection[3] * h)
                        
                        x = int(center_x - width / 2)
                        y = int(center_y - height / 2)
                        
                        boxes.append([x, y, width, height])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Apply NMS
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
            
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    confidence = confidences[i]
                    class_id = class_ids[i]
                    
                    label = self._get_class_label(class_id)
                    object_type = self._get_object_type(label)
                    
                    detection = YOLODetection(
                        object_type=object_type,
                        confidence=confidence,
                        bbox=(x, y, w, h),
                        center=(x + w//2, y + h//2),
                        class_id=class_id,
                        label=label
                    )
                    
                    self._extract_poker_attributes(detection)
                    detections.append(detection)
                    
        except Exception as e:
            logger.error(f"OpenCV DNN detection failed: {e}")
            
        return detections
    
    def _mock_detection(self, image: np.ndarray) -> List[YOLODetection]:
        """Mock detection for testing purposes"""
        h, w = image.shape[:2]
        detections = []
        
        # Simulate detecting some cards
        card_positions = [(150, 450), (250, 450), (350, 200), (450, 200), (550, 200)]
        card_names = ['card_ah', 'card_ks', 'card_qd', 'card_jc', 'card_th']
        
        for i, ((x, y), card_name) in enumerate(zip(card_positions, card_names)):
            if x + 80 < w and y + 120 < h:  # Check bounds
                detection = YOLODetection(
                    object_type=PokerObjectType.CARD,
                    confidence=0.85 + i * 0.02,
                    bbox=(x, y, 80, 120),
                    center=(x + 40, y + 60),
                    class_id=i,
                    label=card_name
                )
                self._extract_poker_attributes(detection)
                detections.append(detection)
        
        # Simulate detecting chips
        chip_positions = [(100, 300), (200, 300), (300, 300)]
        chip_values = [25, 100, 500]
        
        for i, ((x, y), value) in enumerate(zip(chip_positions, chip_values)):
            if x + 30 < w and y + 30 < h:
                detection = YOLODetection(
                    object_type=PokerObjectType.CHIP,
                    confidence=0.90,
                    bbox=(x, y, 30, 30),
                    center=(x + 15, y + 15),
                    class_id=53 + i,
                    label=f'chip_{value}',
                    chip_value=value
                )
                detections.append(detection)
        
        # Simulate detecting players
        player_positions = [(50, 100), (700, 100), (50, 500), (700, 500)]
        for i, (x, y) in enumerate(player_positions):
            if x + 100 < w and y + 100 < h:
                detection = YOLODetection(
                    object_type=PokerObjectType.PLAYER,
                    confidence=0.95,
                    bbox=(x, y, 100, 100),
                    center=(x + 50, y + 50),
                    class_id=62,
                    label='player',
                    player_id=i + 1
                )
                detections.append(detection)
        
        return detections
    
    def _get_output_layers(self) -> List[str]:
        """Get YOLO output layer names"""
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        return output_layers
    
    def _get_class_label(self, class_id: int) -> str:
        """Get class label from class ID"""
        for label, cid in self.poker_classes.items():
            if cid == class_id:
                return label
        return f"unknown_{class_id}"
    
    def _get_object_type(self, label: str) -> PokerObjectType:
        """Determine object type from label"""
        if label.startswith('card_'):
            return PokerObjectType.CARD
        elif label.startswith('chip_'):
            return PokerObjectType.CHIP
        elif label == 'player':
            return PokerObjectType.PLAYER
        elif 'button' in label and any(action in label for action in ['fold', 'call', 'raise', 'check', 'bet']):
            return PokerObjectType.ACTION_BUTTON
        elif label in ['dealer_button', 'small_blind', 'big_blind']:
            return PokerObjectType.BETTING_MARKER
        else:
            return PokerObjectType.CHIP  # Default fallback
    
    def _extract_poker_attributes(self, detection: YOLODetection):
        """Extract poker-specific attributes from detection"""
        label = detection.label
        
        if label.startswith('card_') and len(label) >= 6:
            # Extract rank and suit from card label (e.g., 'card_ah' -> rank='a', suit='h')
            if label != 'card_back':
                rank_suit = label[5:]  # Remove 'card_' prefix
                if len(rank_suit) >= 2:
                    detection.card_rank = rank_suit[0].upper()
                    detection.card_suit = rank_suit[1].lower()
        
        elif label.startswith('chip_'):
            # Extract chip value
            try:
                value_str = label[5:]  # Remove 'chip_' prefix
                detection.chip_value = int(value_str)
            except ValueError:
                detection.chip_value = 0
    
    def _categorize_detections(self, detections: List[YOLODetection]) -> Dict[str, List[YOLODetection]]:
        """Categorize detections by object type"""
        categorized = {
            'cards': [],
            'chips': [],
            'players': [],
            'markers': [],
            'action_buttons': []
        }
        
        for detection in detections:
            if detection.object_type == PokerObjectType.CARD:
                categorized['cards'].append(detection)
            elif detection.object_type == PokerObjectType.CHIP:
                categorized['chips'].append(detection)
            elif detection.object_type == PokerObjectType.PLAYER:
                categorized['players'].append(detection)
            elif detection.object_type == PokerObjectType.BETTING_MARKER:
                categorized['markers'].append(detection)
            elif detection.object_type == PokerObjectType.ACTION_BUTTON:
                categorized['action_buttons'].append(detection)
        
        return categorized
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get detection performance statistics"""
        if not self.detection_times:
            return {}
        
        avg_time = np.mean(self.detection_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        min_time = np.min(self.detection_times)
        max_time = np.max(self.detection_times)
        
        return {
            'average_detection_time': avg_time,
            'fps': fps,
            'min_detection_time': min_time,
            'max_detection_time': max_time,
            'total_frames': self.frame_count,
            'total_time': np.sum(self.detection_times)
        }
    
    def reset_performance_stats(self):
        """Reset performance statistics"""
        self.detection_times = []
        self.frame_count = 0
    
    def visualize_detections(self, image: np.ndarray, detections: PokerGameDetection) -> np.ndarray:
        """
        Visualize detections on the image.
        
        Args:
            image: Original image
            detections: Detection results
            
        Returns:
            Image with detection overlays
        """
        vis_image = image.copy()
        
        # Color mapping for different object types
        colors = {
            PokerObjectType.CARD: (0, 255, 0),      # Green
            PokerObjectType.CHIP: (255, 0, 0),      # Blue
            PokerObjectType.PLAYER: (0, 0, 255),    # Red
            PokerObjectType.BETTING_MARKER: (255, 255, 0),  # Cyan
            PokerObjectType.ACTION_BUTTON: (255, 0, 255)    # Magenta
        }
        
        all_detections = (detections.cards + detections.chips + detections.players + 
                         detections.markers + detections.action_buttons)
        
        for detection in all_detections:
            x, y, w, h = detection.bbox
            color = colors.get(detection.object_type, (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{detection.label} ({detection.confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(vis_image, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            cv2.putText(vis_image, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw center point
            cv2.circle(vis_image, detection.center, 3, color, -1)
        
        # Add performance info
        perf_text = f"Detection time: {detections.detection_time:.3f}s"
        cv2.putText(vis_image, perf_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_image
    
    def save_detection_results(self, detections: PokerGameDetection, filename: str):
        """Save detection results to JSON file"""
        results = {
            'detection_time': detections.detection_time,
            'frame_size': detections.frame_size,
            'cards': [self._detection_to_dict(d) for d in detections.cards],
            'chips': [self._detection_to_dict(d) for d in detections.chips],
            'players': [self._detection_to_dict(d) for d in detections.players],
            'markers': [self._detection_to_dict(d) for d in detections.markers],
            'action_buttons': [self._detection_to_dict(d) for d in detections.action_buttons]
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
    
    def _detection_to_dict(self, detection: YOLODetection) -> dict:
        """Convert detection to dictionary for JSON serialization"""
        return {
            'object_type': detection.object_type.value,
            'confidence': detection.confidence,
            'bbox': detection.bbox,
            'center': detection.center,
            'class_id': detection.class_id,
            'label': detection.label,
            'card_rank': detection.card_rank,
            'card_suit': detection.card_suit,
            'chip_value': detection.chip_value,
            'player_id': detection.player_id
        }

# Utility functions for YOLO model setup
def download_yolo_model(model_url: str, model_path: str) -> bool:
    """Download YOLO model from URL"""
    try:
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Downloaded YOLO model to {model_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download YOLO model: {e}")
        return False

def create_poker_yolo_detector(use_gpu: bool = True) -> YOLOPokerDetector:
    """Factory function to create optimized YOLO detector for poker"""
    return YOLOPokerDetector(use_gpu=use_gpu)