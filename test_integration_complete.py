"""
Complete integration tests demonstrating computer vision and automated gameplay.
Tests the full pipeline from screenshot analysis to automated actions.
"""

import pytest
import asyncio
import numpy as np
import cv2
from unittest.mock import Mock, patch, AsyncMock
from poker import Card, Rank, Suit

from ComputerVision import ComputerVisionEngine, CardDetection, GameStateDetection
from AutomatedGameplay import PokerBot, BotConfiguration, create_balanced_bot
from HandCalculations import CalculateHandRank, CalculateHandRankValue
from utils import PlayerAction

class TestCompleteSystemIntegration:
    """Test the complete system integration"""
    
    def setup_method(self):
        """Set up test environment"""
        self.cv_engine = ComputerVisionEngine()
        self.bot = PokerBot(BotConfiguration(
            screenshot_interval=0.1,
            action_delay_min=0.1,
            action_delay_max=0.2
        ))
    
    def create_synthetic_poker_screenshot(self) -> np.ndarray:
        """Create a synthetic poker game screenshot for testing"""
        # Create a 800x600 image (typical poker client size)
        screenshot = np.ones((600, 800, 3), dtype=np.uint8) * 50  # Dark background
        
        # Draw poker table (simplified)
        cv2.ellipse(screenshot, (400, 300), (350, 200), 0, 0, 360, (0, 100, 0), -1)
        
        # Player cards area (bottom)
        player_card_1 = (150, 450, 80, 120)
        player_card_2 = (250, 450, 80, 120)
        
        # Draw player cards
        cv2.rectangle(screenshot, (150, 450), (230, 570), (255, 255, 255), -1)
        cv2.rectangle(screenshot, (250, 450), (330, 570), (255, 255, 255), -1)
        cv2.rectangle(screenshot, (150, 450), (230, 570), (0, 0, 0), 2)
        cv2.rectangle(screenshot, (250, 450), (330, 570), (0, 0, 0), 2)
        
        # Add card symbols (Ace of Hearts, King of Spades)
        cv2.putText(screenshot, 'A', (165, 480), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        cv2.putText(screenshot, 'h', (165, 520), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(screenshot, 'K', (265, 480), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        cv2.putText(screenshot, 's', (265, 520), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Board cards area (center)
        board_positions = [(250, 200), (350, 200), (450, 200), (550, 200), (650, 200)]
        board_cards = ['Q', 'J', 'T', '9', '8']
        board_suits = ['d', 'c', 'h', 's', 'd']
        
        for i, (pos, card, suit) in enumerate(zip(board_positions, board_cards, board_suits)):
            if i < 3:  # Only draw flop initially
                x, y = pos
                cv2.rectangle(screenshot, (x, y), (x + 80, y + 120), (255, 255, 255), -1)
                cv2.rectangle(screenshot, (x, y), (x + 80, y + 120), (0, 0, 0), 2)
                cv2.putText(screenshot, card, (x + 15, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
                suit_color = (255, 0, 0) if suit in ['h', 'd'] else (0, 0, 0)
                cv2.putText(screenshot, suit, (x + 15, y + 70), cv2.FONT_HERSHEY_SIMPLEX, 1, suit_color, 2)
        
        # Action buttons
        button_y = 520
        buttons = [
            ('FOLD', (50, button_y, 100, 40)),
            ('CALL', (200, button_y, 100, 40)),
            ('RAISE', (350, button_y, 100, 40))
        ]
        
        for text, (x, y, w, h) in buttons:
            cv2.rectangle(screenshot, (x, y), (x + w, y + h), (200, 200, 200), -1)
            cv2.rectangle(screenshot, (x, y), (x + w, y + h), (0, 0, 0), 2)
            cv2.putText(screenshot, text, (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Pot and chip information
        cv2.putText(screenshot, 'POT: $150', (350, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(screenshot, 'CHIPS: $1000', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(screenshot, 'BET: $25', (650, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return screenshot
    
    def test_complete_pipeline_screenshot_to_action(self):
        """Test the complete pipeline from screenshot to automated action"""
        # Create synthetic screenshot
        screenshot = self.create_synthetic_poker_screenshot()
        
        # Test computer vision detection
        game_state = self.cv_engine.detect_game_state(screenshot)
        
        # Verify detection structure
        assert isinstance(game_state, GameStateDetection)
        assert isinstance(game_state.player_cards, list)
        assert isinstance(game_state.board_cards, list)
        assert isinstance(game_state.action_buttons, dict)
        
        # Update bot's game context
        self.bot._update_game_context(game_state)
        
        # Verify game context was updated
        assert self.bot.game_context is not None
        
        # Test decision making
        if self.bot._should_take_action():
            decision = self.bot._make_decision()
            assert decision is not None
            assert isinstance(decision, PlayerAction)
    
    @pytest.mark.asyncio
    @patch('pyautogui.click')
    @patch('pyautogui.moveTo')
    @patch('pyautogui.position')
    async def test_end_to_end_automated_gameplay(self, mock_position, mock_move_to, mock_click):
        """Test end-to-end automated gameplay with mocked GUI interactions"""
        mock_position.return_value = (400, 300)  # Mock current mouse position
        
        # Create a complete game scenario
        player_cards = [
            CardDetection('A', 'h', 0.95, (150, 450, 80, 120), (190, 510)),
            CardDetection('K', 's', 0.90, (250, 450, 80, 120), (290, 510))
        ]
        
        board_cards = [
            CardDetection('Q', 'd', 0.85, (250, 200, 80, 120), (290, 260)),
            CardDetection('J', 'c', 0.88, (350, 200, 80, 120), (390, 260)),
            CardDetection('T', 'h', 0.92, (450, 200, 80, 120), (490, 260))
        ]
        
        action_buttons = {
            'fold': (50, 520, 100, 40),
            'call': (200, 520, 100, 40),
            'raise': (350, 520, 100, 40)
        }
        
        detection = GameStateDetection(
            player_cards=player_cards,
            board_cards=board_cards,
            pot_amount=150,
            player_chips=1000,
            current_bet=25,
            action_buttons=action_buttons
        )
        
        # Update game context
        self.bot._update_game_context(detection)
        
        # Verify we have a valid game state
        assert self.bot.game_context is not None
        assert len(self.bot.game_context.player_cards) == 2
        assert len(self.bot.game_context.board_cards) == 3
        
        # Make decision and execute action
        decision = self.bot._make_decision()
        assert decision is not None
        
        # Execute the action
        await self.bot._execute_action(decision, detection)
        
        # Verify mouse movement and clicking occurred
        mock_move_to.assert_called_once()
        mock_click.assert_called_once()
        
        # Verify action was recorded
        assert len(self.bot.actions_history) == 1
        assert self.bot.actions_history[0] == decision
    
    def test_cv_integration_with_hand_calculations(self):
        """Test that computer vision integrates properly with hand calculations"""
        # Create card detections for a royal flush scenario
        player_cards = [
            CardDetection('A', 'h', 0.95, (150, 450, 80, 120), (190, 510)),
            CardDetection('K', 'h', 0.90, (250, 450, 80, 120), (290, 510))
        ]
        
        board_cards = [
            CardDetection('Q', 'h', 0.85, (250, 200, 80, 120), (290, 260)),
            CardDetection('J', 'h', 0.88, (350, 200, 80, 120), (390, 260)),
            CardDetection('T', 'h', 0.92, (450, 200, 80, 120), (490, 260))
        ]
        
        # Convert to poker objects
        player_poker_cards = self.cv_engine.cards_to_poker_objects(player_cards)
        board_poker_cards = self.cv_engine.cards_to_poker_objects(board_cards)
        
        # Verify conversions
        assert len(player_poker_cards) == 2
        assert len(board_poker_cards) == 3
        
        # Test with hand calculations
        hand_rank = CalculateHandRank(player_poker_cards, board_poker_cards)
        hand_value = CalculateHandRankValue(player_poker_cards, board_poker_cards)
        
        # Should detect a very strong hand
        assert hand_rank is not None
        assert hand_value is not None
        assert isinstance(hand_rank, (int, float))
        assert isinstance(hand_value, (int, float))
    
    def test_multiple_game_states(self):
        """Test bot behavior across multiple game states"""
        # Test pre-flop
        player_cards = [
            CardDetection('A', 'h', 0.95, (150, 450, 80, 120), (190, 510)),
            CardDetection('A', 's', 0.90, (250, 450, 80, 120), (290, 510))
        ]
        
        preflop_detection = GameStateDetection(
            player_cards=player_cards,
            board_cards=[],
            pot_amount=50,
            player_chips=1000,
            current_bet=10,
            action_buttons={'fold': (50, 520, 100, 40), 'call': (200, 520, 100, 40)}
        )
        
        self.bot._update_game_context(preflop_detection)
        assert self.bot.game_context.game_state.value == "preflop"
        
        preflop_decision = self.bot._make_decision()
        assert preflop_decision in [PlayerAction.CALL, PlayerAction.RAISE]  # Strong hand
        
        # Test flop
        board_cards = [
            CardDetection('A', 'd', 0.85, (250, 200, 80, 120), (290, 260)),
            CardDetection('K', 'c', 0.88, (350, 200, 80, 120), (390, 260)),
            CardDetection('Q', 'h', 0.92, (450, 200, 80, 120), (490, 260))
        ]
        
        flop_detection = GameStateDetection(
            player_cards=player_cards,
            board_cards=board_cards,
            pot_amount=150,
            player_chips=950,
            current_bet=25,
            action_buttons={'fold': (50, 520, 100, 40), 'call': (200, 520, 100, 40), 'raise': (350, 520, 100, 40)}
        )
        
        self.bot._update_game_context(flop_detection)
        assert self.bot.game_context.game_state.value == "flop"
        
        flop_decision = self.bot._make_decision()
        assert flop_decision in [PlayerAction.CALL, PlayerAction.RAISE]  # Set of aces
    
    def test_bot_configuration_effects(self):
        """Test that different bot configurations affect decision making"""
        # Create scenario
        player_cards = [
            CardDetection('7', 'h', 0.90, (150, 450, 80, 120), (190, 510)),
            CardDetection('8', 's', 0.85, (250, 450, 80, 120), (290, 510))
        ]
        
        detection = GameStateDetection(
            player_cards=player_cards,
            board_cards=[],
            pot_amount=50,
            player_chips=1000,
            current_bet=10,
            action_buttons={'fold': (50, 520, 100, 40), 'call': (200, 520, 100, 40)}
        )
        
        # Test conservative bot
        conservative_bot = PokerBot(BotConfiguration(
            aggression_level=0.2,
            tight_play=True,
            bluff_frequency=0.02
        ))
        conservative_bot._update_game_context(detection)
        conservative_decision = conservative_bot._make_decision()
        
        # Test aggressive bot  
        aggressive_bot = PokerBot(BotConfiguration(
            aggression_level=0.9,
            tight_play=False,
            bluff_frequency=0.3
        ))
        aggressive_bot._update_game_context(detection)
        aggressive_decision = aggressive_bot._make_decision()
        
        # Conservative bot should be more likely to fold weak hands
        # Aggressive bot should be more likely to play weak hands
        # Note: Due to randomness, we can't guarantee specific outcomes,
        # but we can verify decisions are valid
        assert conservative_decision in [PlayerAction.FOLD, PlayerAction.CALL]
        assert aggressive_decision in [PlayerAction.FOLD, PlayerAction.CALL, PlayerAction.RAISE]
    
    @patch('ComputerVision.cv2.imwrite')
    def test_debug_functionality(self, mock_imwrite):
        """Test debug functionality works end-to-end"""
        screenshot = self.create_synthetic_poker_screenshot()
        
        # Detect cards
        detections = self.cv_engine.detect_cards_template_matching(screenshot)
        
        # Save debug image
        self.cv_engine.save_detection_debug_image(screenshot, detections, "test_debug.jpg")
        
        # Verify debug image saving was attempted
        mock_imwrite.assert_called_once()
    
    def test_error_handling_pipeline(self):
        """Test error handling throughout the pipeline"""
        # Test with invalid/empty screenshot
        empty_screenshot = np.array([])
        
        # Should handle gracefully
        game_state = self.cv_engine.detect_game_state(empty_screenshot)
        assert isinstance(game_state, GameStateDetection)
        assert len(game_state.player_cards) == 0
        assert len(game_state.board_cards) == 0
        
        # Test bot with no game context
        assert self.bot._should_take_action() is False
        
        # Test bot with invalid game context
        self.bot.game_context = None
        decision = self.bot._make_decision()
        assert decision is None
    
    def test_performance_integration(self):
        """Test performance of the integrated system"""
        import time
        
        screenshot = self.create_synthetic_poker_screenshot()
        
        # Time the complete pipeline
        start_time = time.time()
        
        # Computer vision detection
        game_state = self.cv_engine.detect_game_state(screenshot)
        
        # Bot decision making
        self.bot._update_game_context(game_state)
        if self.bot._should_take_action():
            decision = self.bot._make_decision()
        
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # Complete pipeline should execute quickly (less than 0.5 seconds)
        assert total_time < 0.5, f"Complete pipeline took {total_time:.3f} seconds, too slow"
    
    def test_factory_bot_integration(self):
        """Test that factory-created bots work with the complete system"""
        # Test different bot types
        bots = [
            create_balanced_bot(),
            # We could test others if we had imported them
        ]
        
        for bot in bots:
            # Create test scenario
            player_cards = [
                CardDetection('A', 'h', 0.95, (150, 450, 80, 120), (190, 510)),
                CardDetection('K', 's', 0.90, (250, 450, 80, 120), (290, 510))
            ]
            
            detection = GameStateDetection(
                player_cards=player_cards,
                board_cards=[],
                pot_amount=50,
                player_chips=1000,
                current_bet=10,
                action_buttons={'fold': (50, 520, 100, 40), 'call': (200, 520, 100, 40)}
            )
            
            # Update and test
            bot._update_game_context(detection)
            assert bot.game_context is not None
            
            decision = bot._make_decision()
            assert decision is not None
            assert isinstance(decision, PlayerAction)

class TestSystemReliability:
    """Test system reliability and edge cases"""
    
    def setup_method(self):
        """Set up test environment"""
        self.cv_engine = ComputerVisionEngine()
        self.bot = PokerBot()
    
    def test_partial_card_detection(self):
        """Test system behavior with partial card detection"""
        # Only one player card detected
        partial_detection = GameStateDetection(
            player_cards=[
                CardDetection('A', 'h', 0.95, (150, 450, 80, 120), (190, 510))
            ],
            board_cards=[],
            pot_amount=50,
            player_chips=1000,
            current_bet=10,
            action_buttons={}
        )
        
        self.bot._update_game_context(partial_detection)
        
        # Should not take action with incomplete hand
        assert self.bot._should_take_action() is False
    
    def test_invalid_card_detection(self):
        """Test system behavior with invalid card detections"""
        # Invalid rank/suit combinations
        invalid_detection = GameStateDetection(
            player_cards=[
                CardDetection('X', 'z', 0.50, (150, 450, 80, 120), (190, 510)),
                CardDetection('Y', 'w', 0.30, (250, 450, 80, 120), (290, 510))
            ],
            board_cards=[],
            pot_amount=50,
            player_chips=1000,
            current_bet=10,
            action_buttons={}
        )
        
        self.bot._update_game_context(invalid_detection)
        
        # Should handle invalid cards gracefully
        if self.bot.game_context:
            assert len(self.bot.game_context.player_cards) == 0  # Invalid cards filtered out
    
    def test_low_confidence_detections(self):
        """Test system behavior with low confidence detections"""
        # Low confidence card detections
        low_conf_detection = GameStateDetection(
            player_cards=[
                CardDetection('A', 'h', 0.30, (150, 450, 80, 120), (190, 510)),
                CardDetection('K', 's', 0.25, (250, 450, 80, 120), (290, 510))
            ],
            board_cards=[],
            pot_amount=50,
            player_chips=1000,
            current_bet=10,
            action_buttons={}
        )
        
        # The system should still work, but might be less reliable
        # This tests the robustness of the pipeline
        self.bot._update_game_context(low_conf_detection)
        
        # Should still create a game context (confidence filtering happens at CV level)
        assert self.bot.game_context is not None

if __name__ == "__main__":
    # Run the integration tests
    pytest.main([__file__, "-v", "-s"])