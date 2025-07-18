"""
Computer Vision module for poker card detection and game state recognition.
Supports real-time screenshot analysis and card identification.
"""

import cv2
import numpy as np
import pyautogui
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import os
import json
from dataclasses import dataclass
from poker import Card, Rank, Suit
from utils import RANKS, SUITS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CardDetection:
    """Represents a detected card with its properties"""
    rank: str
    suit: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    center: Tuple[int, int]

@dataclass
class GameStateDetection:
    """Represents the detected game state"""
    player_cards: List[CardDetection]
    board_cards: List[CardDetection]
    pot_amount: Optional[int]
    player_chips: Optional[int]
    current_bet: Optional[int]
    action_buttons: Dict[str, Tuple[int, int, int, int]]  # button_name -> bbox

class CardRecognitionCNN(nn.Module):
    """Convolutional Neural Network for card recognition"""
    
    def __init__(self, num_ranks=13, num_suits=4):
        super(CardRecognitionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        # Calculate the size of the flattened features
        # Assuming input size of 64x64
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.rank_head = nn.Linear(512, num_ranks)
        self.suit_head = nn.Linear(512, num_suits)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        rank_logits = self.rank_head(x)
        suit_logits = self.suit_head(x)
        
        return rank_logits, suit_logits

class ComputerVisionEngine:
    """Main computer vision engine for poker game analysis"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.card_model = CardRecognitionCNN()
        
        # Load pre-trained model if available
        if model_path and os.path.exists(model_path):
            self.card_model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Loaded card recognition model from {model_path}")
        else:
            logger.warning("No pre-trained model found. Using template matching fallback.")
            
        self.card_model.to(self.device)
        self.card_model.eval()
        
        # Template matching setup for fallback
        self.rank_templates = self._load_rank_templates()
        self.suit_templates = self._load_suit_templates()
        
        # Screen capture settings
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1
        
    def _load_rank_templates(self) -> Dict[str, np.ndarray]:
        """Load rank templates for template matching"""
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        templates = {}
        
        # Create simple text-based templates (in a real implementation, 
        # these would be loaded from image files)
        for rank in ranks:
            # Create a simple template (placeholder implementation)
            template = np.ones((30, 20), dtype=np.uint8) * 255
            cv2.putText(template, rank, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
            templates[rank] = template
            
        return templates
    
    def _load_suit_templates(self) -> Dict[str, np.ndarray]:
        """Load suit templates for template matching"""
        suits = ['c', 'd', 'h', 's']  # clubs, diamonds, hearts, spades
        templates = {}
        
        for suit in suits:
            # Create simple suit templates (placeholder implementation)
            template = np.ones((30, 20), dtype=np.uint8) * 255
            cv2.putText(template, suit, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
            templates[suit] = template
            
        return templates
    
    def capture_screen(self, region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Capture screenshot of the screen or specified region"""
        try:
            if region:
                screenshot = pyautogui.screenshot(region=region)
            else:
                screenshot = pyautogui.screenshot()
            
            # Convert PIL image to OpenCV format
            opencv_image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            return opencv_image
        except Exception as e:
            logger.error(f"Failed to capture screen: {e}")
            return np.array([])
    
    def detect_cards_template_matching(self, image: np.ndarray) -> List[CardDetection]:
        """Detect cards using template matching (fallback method)"""
        cards = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Define card regions to search (these would be calibrated for specific poker applications)
        card_regions = [
            (100, 400, 80, 120),  # Player card 1
            (200, 400, 80, 120),  # Player card 2
            (250, 200, 80, 120),  # Flop card 1
            (350, 200, 80, 120),  # Flop card 2
            (450, 200, 80, 120),  # Flop card 3
            (550, 200, 80, 120),  # Turn card
            (650, 200, 80, 120),  # River card
        ]
        
        for i, (x, y, w, h) in enumerate(card_regions):
            if x + w >= image.shape[1] or y + h >= image.shape[0]:
                continue
                
            card_roi = gray[y:y+h, x:x+w]
            
            # Detect rank and suit using template matching
            rank, rank_conf = self._match_rank(card_roi)
            suit, suit_conf = self._match_suit(card_roi)
            
            if rank and suit and rank_conf > 0.6 and suit_conf > 0.6:
                detection = CardDetection(
                    rank=rank,
                    suit=suit,
                    confidence=min(rank_conf, suit_conf),
                    bbox=(x, y, w, h),
                    center=(x + w//2, y + h//2)
                )
                cards.append(detection)
                
        return cards
    
    def _match_rank(self, card_roi: np.ndarray) -> Tuple[Optional[str], float]:
        """Match rank using template matching"""
        best_match = None
        best_score = 0
        
        for rank, template in self.rank_templates.items():
            if template.shape[0] > card_roi.shape[0] or template.shape[1] > card_roi.shape[1]:
                continue
                
            result = cv2.matchTemplate(card_roi, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            if max_val > best_score:
                best_score = max_val
                best_match = rank
                
        return best_match, best_score
    
    def _match_suit(self, card_roi: np.ndarray) -> Tuple[Optional[str], float]:
        """Match suit using template matching"""
        best_match = None
        best_score = 0
        
        for suit, template in self.suit_templates.items():
            if template.shape[0] > card_roi.shape[0] or template.shape[1] > card_roi.shape[1]:
                continue
                
            result = cv2.matchTemplate(card_roi, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            if max_val > best_score:
                best_score = max_val
                best_match = suit
                
        return best_match, best_score
    
    def detect_cards_cnn(self, image: np.ndarray) -> List[CardDetection]:
        """Detect cards using CNN (when model is available)"""
        cards = []
        
        # First, detect card regions using traditional CV
        card_regions = self._detect_card_regions(image)
        
        for region in card_regions:
            x, y, w, h = region
            card_roi = image[y:y+h, x:x+w]
            
            # Preprocess for CNN
            card_tensor = self._preprocess_card_for_cnn(card_roi)
            
            with torch.no_grad():
                rank_logits, suit_logits = self.card_model(card_tensor)
                
                rank_probs = F.softmax(rank_logits, dim=1)
                suit_probs = F.softmax(suit_logits, dim=1)
                
                rank_idx = torch.argmax(rank_probs).item()
                suit_idx = torch.argmax(suit_probs).item()
                
                rank_conf = rank_probs[0, rank_idx].item()
                suit_conf = suit_probs[0, suit_idx].item()
                
                if rank_conf > 0.7 and suit_conf > 0.7:
                    rank_names = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
                    suit_names = ['c', 'd', 'h', 's']
                    
                    detection = CardDetection(
                        rank=rank_names[rank_idx],
                        suit=suit_names[suit_idx],
                        confidence=min(rank_conf, suit_conf),
                        bbox=(x, y, w, h),
                        center=(x + w//2, y + h//2)
                    )
                    cards.append(detection)
        
        return cards
    
    def _detect_card_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect potential card regions in the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        card_regions = []
        for contour in contours:
            # Calculate contour area and bounding rectangle
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if the aspect ratio is card-like
                aspect_ratio = w / h
                if 0.6 <= aspect_ratio <= 0.8:  # Typical card aspect ratio
                    card_regions.append((x, y, w, h))
        
        return card_regions
    
    def _preprocess_card_for_cnn(self, card_roi: np.ndarray) -> torch.Tensor:
        """Preprocess card ROI for CNN input"""
        # Resize to expected input size
        resized = cv2.resize(card_roi, (64, 64))
        
        # Convert to RGB and normalize
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def detect_game_state(self, image: np.ndarray) -> GameStateDetection:
        """Detect complete game state from image"""
        # Detect cards
        cards = self.detect_cards_template_matching(image)
        
        # Separate player cards and board cards based on position
        player_cards = []
        board_cards = []
        
        for card in cards:
            center_x, center_y = card.center
            if center_y > image.shape[0] * 0.7:  # Bottom part of screen
                player_cards.append(card)
            else:  # Middle part of screen
                board_cards.append(card)
        
        # Detect action buttons (simplified implementation)
        action_buttons = self._detect_action_buttons(image)
        
        # Detect text-based information (pot, chips, etc.)
        pot_amount = self._detect_pot_amount(image)
        player_chips = self._detect_player_chips(image)
        current_bet = self._detect_current_bet(image)
        
        return GameStateDetection(
            player_cards=player_cards,
            board_cards=board_cards,
            pot_amount=pot_amount,
            player_chips=player_chips,
            current_bet=current_bet,
            action_buttons=action_buttons
        )
    
    def _detect_action_buttons(self, image: np.ndarray) -> Dict[str, Tuple[int, int, int, int]]:
        """Detect action buttons in the image"""
        # This is a simplified implementation
        # In a real application, you would train specific detectors for each button
        buttons = {}
        
        # Common button regions (these would be calibrated for specific poker applications)
        button_regions = {
            'fold': (50, 500, 100, 40),
            'call': (200, 500, 100, 40),
            'raise': (350, 500, 100, 40),
            'check': (200, 500, 100, 40),
            'bet': (350, 500, 100, 40),
        }
        
        return button_regions
    
    def _detect_pot_amount(self, image: np.ndarray) -> Optional[int]:
        """Detect pot amount using OCR"""
        # Simplified implementation - would use OCR in practice
        return None
    
    def _detect_player_chips(self, image: np.ndarray) -> Optional[int]:
        """Detect player chip count using OCR"""
        # Simplified implementation - would use OCR in practice
        return None
    
    def _detect_current_bet(self, image: np.ndarray) -> Optional[int]:
        """Detect current bet amount using OCR"""
        # Simplified implementation - would use OCR in practice
        return None
    
    def cards_to_poker_objects(self, card_detections: List[CardDetection]) -> List[Card]:
        """Convert card detections to poker library Card objects"""
        poker_cards = []
        
        rank_mapping = {
            '2': Rank.DEUCE, '3': Rank.THREE, '4': Rank.FOUR, '5': Rank.FIVE,
            '6': Rank.SIX, '7': Rank.SEVEN, '8': Rank.EIGHT, '9': Rank.NINE,
            'T': Rank.TEN, 'J': Rank.JACK, 'Q': Rank.QUEEN, 'K': Rank.KING, 'A': Rank.ACE
        }
        
        suit_mapping = {
            'c': Suit.CLUBS, 'd': Suit.DIAMONDS, 'h': Suit.HEARTS, 's': Suit.SPADES
        }
        
        for detection in card_detections:
            if detection.rank in rank_mapping and detection.suit in suit_mapping:
                card = Card(rank_mapping[detection.rank], suit_mapping[detection.suit])
                poker_cards.append(card)
        
        return poker_cards
    
    def save_detection_debug_image(self, image: np.ndarray, detections: List[CardDetection], 
                                 filename: str = "debug_detection.jpg"):
        """Save image with detection overlays for debugging"""
        debug_image = image.copy()
        
        for detection in detections:
            x, y, w, h = detection.bbox
            cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            label = f"{detection.rank}{detection.suit} ({detection.confidence:.2f})"
            cv2.putText(debug_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 1)
        
        cv2.imwrite(filename, debug_image)
        logger.info(f"Debug image saved as {filename}")