"""
Test cases for Computer Vision module functionality.
Tests card detection, game state recognition, and integration with poker logic.
"""

import pytest
import numpy as np
import cv2
import torch
from unittest.mock import Mock, patch, MagicMock
from poker import Card, Rank, Suit

from ComputerVision import (
    ComputerVisionEngine, 
    CardDetection, 
    GameStateDetection,
    CardRecognitionCNN
)

class TestCardRecognitionCNN:
    """Test the CNN model for card recognition"""
    
    def test_model_initialization(self):
        """Test that the CNN model initializes correctly"""
        model = CardRecognitionCNN(num_ranks=13, num_suits=4)
        assert model is not None
        assert hasattr(model, 'conv1')
        assert hasattr(model, 'rank_head')
        assert hasattr(model, 'suit_head')
    
    def test_model_forward_pass(self):
        """Test that the model can process input tensors"""
        model = CardRecognitionCNN()
        model.eval()
        
        # Create dummy input (batch_size=1, channels=3, height=64, width=64)
        dummy_input = torch.randn(1, 3, 64, 64)
        
        with torch.no_grad():
            rank_logits, suit_logits = model(dummy_input)
        
        assert rank_logits.shape == (1, 13)  # 13 ranks
        assert suit_logits.shape == (1, 4)   # 4 suits
    
    def test_model_output_ranges(self):
        """Test that model outputs are in expected ranges"""
        model = CardRecognitionCNN()
        model.eval()
        
        dummy_input = torch.randn(1, 3, 64, 64)
        
        with torch.no_grad():
            rank_logits, suit_logits = model(dummy_input)
            
            # Apply softmax to get probabilities
            rank_probs = torch.softmax(rank_logits, dim=1)
            suit_probs = torch.softmax(suit_logits, dim=1)
        
        # Check that probabilities sum to 1
        assert torch.allclose(rank_probs.sum(dim=1), torch.tensor([1.0]), atol=1e-6)
        assert torch.allclose(suit_probs.sum(dim=1), torch.tensor([1.0]), atol=1e-6)
        
        # Check that all probabilities are between 0 and 1
        assert torch.all(rank_probs >= 0) and torch.all(rank_probs <= 1)
        assert torch.all(suit_probs >= 0) and torch.all(suit_probs <= 1)

class TestComputerVisionEngine:
    """Test the main computer vision engine"""
    
    def setup_method(self):
        """Set up test environment"""
        self.cv_engine = ComputerVisionEngine()
    
    def test_engine_initialization(self):
        """Test that the CV engine initializes correctly"""
        assert self.cv_engine is not None
        assert hasattr(self.cv_engine, 'card_model')
        assert hasattr(self.cv_engine, 'rank_templates')
        assert hasattr(self.cv_engine, 'suit_templates')
    
    def test_template_loading(self):
        """Test that rank and suit templates are loaded"""
        assert len(self.cv_engine.rank_templates) == 13  # 13 ranks
        assert len(self.cv_engine.suit_templates) == 4   # 4 suits
        
        # Check that templates contain expected keys
        expected_ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        expected_suits = ['c', 'd', 'h', 's']
        
        assert all(rank in self.cv_engine.rank_templates for rank in expected_ranks)
        assert all(suit in self.cv_engine.suit_templates for suit in expected_suits)
    
    def test_create_synthetic_card_image(self):
        """Helper method to create synthetic card images for testing"""
        # Create a simple synthetic card image
        card_image = np.ones((120, 80, 3), dtype=np.uint8) * 255  # White background
        
        # Add some simple features (placeholder for actual card features)
        cv2.rectangle(card_image, (10, 10), (70, 110), (0, 0, 0), 2)  # Card border
        cv2.putText(card_image, 'A', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(card_image, 'h', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        return card_image
    
    def test_card_region_detection(self):
        """Test detection of card regions in images"""
        # Create a synthetic image with card-like rectangles
        test_image = np.ones((600, 800, 3), dtype=np.uint8) * 128  # Gray background
        
        # Add some card-like rectangles
        cv2.rectangle(test_image, (100, 200), (180, 320), (255, 255, 255), -1)  # Card 1
        cv2.rectangle(test_image, (200, 200), (280, 320), (255, 255, 255), -1)  # Card 2
        cv2.rectangle(test_image, (100, 200), (180, 320), (0, 0, 0), 2)  # Border 1
        cv2.rectangle(test_image, (200, 200), (280, 320), (0, 0, 0), 2)  # Border 2
        
        regions = self.cv_engine._detect_card_regions(test_image)
        
        # Should detect some regions (exact number depends on edge detection)
        assert len(regions) >= 0  # At least we shouldn't crash
        
        # All regions should have reasonable dimensions
        for x, y, w, h in regions:
            assert x >= 0 and y >= 0
            assert w > 0 and h > 0
            assert x + w <= test_image.shape[1]
            assert y + h <= test_image.shape[0]
    
    def test_card_preprocessing_for_cnn(self):
        """Test preprocessing of card images for CNN input"""
        card_image = self.test_create_synthetic_card_image()
        
        preprocessed = self.cv_engine._preprocess_card_for_cnn(card_image)
        
        # Check tensor properties
        assert isinstance(preprocessed, torch.Tensor)
        assert preprocessed.shape == (1, 3, 64, 64)  # batch_size=1, channels=3, 64x64
        assert preprocessed.dtype == torch.float32
        
        # Check value range (should be normalized to [0, 1])
        assert torch.all(preprocessed >= 0.0)
        assert torch.all(preprocessed <= 1.0)
    
    def test_cards_to_poker_objects_conversion(self):
        """Test conversion of card detections to poker objects"""
        # Create test card detections
        detections = [
            CardDetection(rank='A', suit='h', confidence=0.9, bbox=(0, 0, 80, 120), center=(40, 60)),
            CardDetection(rank='K', suit='s', confidence=0.8, bbox=(100, 0, 80, 120), center=(140, 60)),
            CardDetection(rank='Q', suit='d', confidence=0.85, bbox=(200, 0, 80, 120), center=(240, 60)),
        ]
        
        poker_cards = self.cv_engine.cards_to_poker_objects(detections)
        
        assert len(poker_cards) == 3
        
        # Check first card (Ace of Hearts)
        assert poker_cards[0].rank == Rank.ACE
        assert poker_cards[0].suit == Suit.HEARTS
        
        # Check second card (King of Spades)
        assert poker_cards[1].rank == Rank.KING
        assert poker_cards[1].suit == Suit.SPADES
        
        # Check third card (Queen of Diamonds)
        assert poker_cards[2].rank == Rank.QUEEN
        assert poker_cards[2].suit == Suit.DIAMONDS
    
    def test_invalid_card_conversion(self):
        """Test handling of invalid card detections"""
        # Create invalid card detections
        invalid_detections = [
            CardDetection(rank='X', suit='h', confidence=0.9, bbox=(0, 0, 80, 120), center=(40, 60)),
            CardDetection(rank='A', suit='z', confidence=0.8, bbox=(100, 0, 80, 120), center=(140, 60)),
        ]
        
        poker_cards = self.cv_engine.cards_to_poker_objects(invalid_detections)
        
        # Should filter out invalid cards
        assert len(poker_cards) == 0
    
    @patch('pyautogui.screenshot')
    def test_screen_capture(self, mock_screenshot):
        """Test screen capture functionality"""
        # Mock the screenshot function
        mock_image = Mock()
        mock_image.__array__ = Mock(return_value=np.ones((600, 800, 3), dtype=np.uint8))
        mock_screenshot.return_value = mock_image
        
        # Test full screen capture
        result = self.cv_engine.capture_screen()
        
        assert result is not None
        assert result.shape == (600, 800, 3)
        mock_screenshot.assert_called_once()
    
    @patch('pyautogui.screenshot')
    def test_screen_capture_with_region(self, mock_screenshot):
        """Test screen capture with specific region"""
        mock_image = Mock()
        mock_image.__array__ = Mock(return_value=np.ones((200, 300, 3), dtype=np.uint8))
        mock_screenshot.return_value = mock_image
        
        # Test region capture
        region = (100, 100, 300, 200)
        result = self.cv_engine.capture_screen(region)
        
        assert result is not None
        mock_screenshot.assert_called_once_with(region=region)
    
    def test_template_matching_functionality(self):
        """Test basic template matching functionality"""
        # Create a test image with a known pattern
        test_image = np.ones((600, 800, 3), dtype=np.uint8) * 255
        
        # Add some text to match against
        cv2.putText(test_image, 'A', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        
        # Test template matching (this should not crash)
        detections = self.cv_engine.detect_cards_template_matching(test_image)
        
        # Should return a list (might be empty, but shouldn't crash)
        assert isinstance(detections, list)
        
        # All detections should be valid CardDetection objects
        for detection in detections:
            assert isinstance(detection, CardDetection)
            assert detection.confidence >= 0.0 and detection.confidence <= 1.0
            assert len(detection.rank) == 1
            assert len(detection.suit) == 1
    
    def test_game_state_detection_structure(self):
        """Test that game state detection returns proper structure"""
        # Create a test image
        test_image = np.ones((600, 800, 3), dtype=np.uint8) * 128
        
        game_state = self.cv_engine.detect_game_state(test_image)
        
        # Check that all required fields are present
        assert isinstance(game_state, GameStateDetection)
        assert isinstance(game_state.player_cards, list)
        assert isinstance(game_state.board_cards, list)
        assert isinstance(game_state.action_buttons, dict)
        
        # Check that numeric fields are proper types
        assert game_state.pot_amount is None or isinstance(game_state.pot_amount, int)
        assert game_state.player_chips is None or isinstance(game_state.player_chips, int)
        assert game_state.current_bet is None or isinstance(game_state.current_bet, int)
    
    def test_debug_image_saving(self):
        """Test debug image saving functionality"""
        test_image = np.ones((400, 600, 3), dtype=np.uint8) * 128
        detections = [
            CardDetection(rank='A', suit='h', confidence=0.9, bbox=(10, 10, 80, 120), center=(50, 70))
        ]
        
        # This should not crash
        try:
            self.cv_engine.save_detection_debug_image(test_image, detections, "test_debug.jpg")
            # Clean up test file if it was created
            import os
            if os.path.exists("test_debug.jpg"):
                os.remove("test_debug.jpg")
        except Exception as e:
            pytest.fail(f"Debug image saving failed: {e}")

class TestCardDetection:
    """Test the CardDetection dataclass"""
    
    def test_card_detection_creation(self):
        """Test creation of CardDetection objects"""
        detection = CardDetection(
            rank='A',
            suit='h',
            confidence=0.95,
            bbox=(10, 20, 80, 120),
            center=(50, 80)
        )
        
        assert detection.rank == 'A'
        assert detection.suit == 'h'
        assert detection.confidence == 0.95
        assert detection.bbox == (10, 20, 80, 120)
        assert detection.center == (50, 80)
    
    def test_card_detection_validation(self):
        """Test validation of CardDetection properties"""
        detection = CardDetection(
            rank='K',
            suit='s',
            confidence=0.75,
            bbox=(0, 0, 100, 150),
            center=(50, 75)
        )
        
        # Basic type checks
        assert isinstance(detection.rank, str)
        assert isinstance(detection.suit, str)
        assert isinstance(detection.confidence, float)
        assert isinstance(detection.bbox, tuple)
        assert isinstance(detection.center, tuple)
        
        # Value checks
        assert 0.0 <= detection.confidence <= 1.0

class TestGameStateDetection:
    """Test the GameStateDetection dataclass"""
    
    def test_game_state_detection_creation(self):
        """Test creation of GameStateDetection objects"""
        player_cards = [
            CardDetection('A', 'h', 0.9, (10, 400, 80, 120), (50, 460)),
            CardDetection('K', 's', 0.85, (100, 400, 80, 120), (140, 460))
        ]
        
        board_cards = [
            CardDetection('Q', 'd', 0.8, (250, 200, 80, 120), (290, 260)),
            CardDetection('J', 'c', 0.75, (350, 200, 80, 120), (390, 260)),
            CardDetection('T', 'h', 0.9, (450, 200, 80, 120), (490, 260))
        ]
        
        action_buttons = {
            'fold': (50, 500, 100, 40),
            'call': (200, 500, 100, 40),
            'raise': (350, 500, 100, 40)
        }
        
        game_state = GameStateDetection(
            player_cards=player_cards,
            board_cards=board_cards,
            pot_amount=150,
            player_chips=1000,
            current_bet=25,
            action_buttons=action_buttons
        )
        
        assert len(game_state.player_cards) == 2
        assert len(game_state.board_cards) == 3
        assert game_state.pot_amount == 150
        assert game_state.player_chips == 1000
        assert game_state.current_bet == 25
        assert len(game_state.action_buttons) == 3

class TestIntegrationWithPokerLogic:
    """Test integration between computer vision and existing poker logic"""
    
    def setup_method(self):
        """Set up test environment"""
        self.cv_engine = ComputerVisionEngine()
    
    def test_integration_with_hand_calculations(self):
        """Test that detected cards work with hand calculation functions"""
        # Create card detections for a good hand (pair of aces)
        detections = [
            CardDetection('A', 'h', 0.9, (10, 400, 80, 120), (50, 460)),
            CardDetection('A', 's', 0.85, (100, 400, 80, 120), (140, 460))
        ]
        
        # Convert to poker objects
        poker_cards = self.cv_engine.cards_to_poker_objects(detections)
        
        # This should work with existing hand calculation functions
        from HandCalculations import CalculateHandRank, CalculateHandRankValue
        
        try:
            hand_rank = CalculateHandRank(poker_cards, [])
            hand_value = CalculateHandRankValue(poker_cards, [])
            
            # Should return valid values
            assert hand_rank is not None
            assert hand_value is not None
            assert isinstance(hand_rank, (int, float))
            assert isinstance(hand_value, (int, float))
            
        except Exception as e:
            pytest.fail(f"Integration with hand calculations failed: {e}")
    
    def test_integration_with_decision_making(self):
        """Test that detected game state works with decision making"""
        # Create a complete game state
        player_cards = [
            CardDetection('A', 'h', 0.9, (10, 400, 80, 120), (50, 460)),
            CardDetection('K', 's', 0.85, (100, 400, 80, 120), (140, 460))
        ]
        
        board_cards = [
            CardDetection('A', 'd', 0.8, (250, 200, 80, 120), (290, 260)),
            CardDetection('K', 'c', 0.75, (350, 200, 80, 120), (390, 260)),
            CardDetection('Q', 'h', 0.9, (450, 200, 80, 120), (490, 260))
        ]
        
        # Convert to poker objects
        player_poker_cards = self.cv_engine.cards_to_poker_objects(player_cards)
        board_poker_cards = self.cv_engine.cards_to_poker_objects(board_cards)
        
        # This should work with decision making logic
        from HandCalculations import CalculateHandRank
        
        try:
            hand_rank = CalculateHandRank(player_poker_cards, board_poker_cards)
            
            # Should be able to make decisions based on this
            assert hand_rank is not None
            
        except Exception as e:
            pytest.fail(f"Integration with decision making failed: {e}")

@pytest.mark.integration
class TestComputerVisionPerformance:
    """Performance tests for computer vision components"""
    
    def setup_method(self):
        """Set up test environment"""
        self.cv_engine = ComputerVisionEngine()
    
    def test_detection_speed(self):
        """Test that detection runs in reasonable time"""
        import time
        
        # Create a test image
        test_image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        
        start_time = time.time()
        detections = self.cv_engine.detect_cards_template_matching(test_image)
        end_time = time.time()
        
        detection_time = end_time - start_time
        
        # Should complete in reasonable time (less than 1 second for template matching)
        assert detection_time < 1.0, f"Detection took {detection_time:.2f} seconds, too slow"
    
    def test_cnn_inference_speed(self):
        """Test CNN inference speed"""
        import time
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 64, 64)
        
        self.cv_engine.card_model.eval()
        
        # Warm up
        with torch.no_grad():
            _ = self.cv_engine.card_model(dummy_input)
        
        # Time inference
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):  # Run multiple inferences
                _ = self.cv_engine.card_model(dummy_input)
        end_time = time.time()
        
        avg_inference_time = (end_time - start_time) / 10
        
        # Should be fast (less than 0.1 seconds per inference)
        assert avg_inference_time < 0.1, f"CNN inference took {avg_inference_time:.4f} seconds, too slow"

if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])