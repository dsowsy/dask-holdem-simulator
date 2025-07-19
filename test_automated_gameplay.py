"""
Test cases for Automated Gameplay module functionality.
Tests bot behavior, decision making, and automation components.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from poker import Card, Rank, Suit

from AutomatedGameplay import (
    PokerBot, 
    BotConfiguration, 
    GameContext, 
    GameState,
    PokerBotTrainer,
    create_conservative_bot,
    create_aggressive_bot,
    create_balanced_bot
)
from ComputerVision import CardDetection, GameStateDetection
from utils import PlayerAction, PlayerPosition

class TestBotConfiguration:
    """Test bot configuration functionality"""
    
    def test_default_configuration(self):
        """Test default bot configuration values"""
        config = BotConfiguration()
        
        assert config.aggression_level == 0.5
        assert config.bluff_frequency == 0.1
        assert config.tight_play is True
        assert config.position_awareness is True
        assert config.bankroll_management is True
        assert config.max_buy_in_percentage == 0.05
        assert config.screenshot_interval == 0.5
        assert config.confidence_threshold == 0.7
        assert config.action_delay_min == 1.0
        assert config.action_delay_max == 3.0
        assert config.mouse_movement_speed == 0.3
    
    def test_custom_configuration(self):
        """Test custom bot configuration"""
        config = BotConfiguration(
            aggression_level=0.8,
            bluff_frequency=0.2,
            tight_play=False,
            screenshot_interval=0.3
        )
        
        assert config.aggression_level == 0.8
        assert config.bluff_frequency == 0.2
        assert config.tight_play is False
        assert config.screenshot_interval == 0.3
        # Other values should remain default
        assert config.bankroll_management is True

class TestGameContext:
    """Test game context functionality"""
    
    def test_game_context_creation(self):
        """Test creation of game context"""
        player_cards = [
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.KING, Suit.SPADES)
        ]
        
        board_cards = [
            Card(Rank.QUEEN, Suit.DIAMONDS),
            Card(Rank.JACK, Suit.CLUBS),
            Card(Rank.TEN, Suit.HEARTS)
        ]
        
        context = GameContext(
            game_state=GameState.FLOP,
            player_cards=player_cards,
            board_cards=board_cards,
            pot_size=150,
            player_chips=1000,
            current_bet=25,
            position=PlayerPosition.DEALER,
            num_players=6,
            actions_taken=[PlayerAction.CALL]
        )
        
        assert context.game_state == GameState.FLOP
        assert len(context.player_cards) == 2
        assert len(context.board_cards) == 3
        assert context.pot_size == 150
        assert context.player_chips == 1000
        assert context.current_bet == 25
        assert context.position == PlayerPosition.DEALER
        assert context.num_players == 6
        assert len(context.actions_taken) == 1

class TestPokerBot:
    """Test the main poker bot functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.config = BotConfiguration(
            screenshot_interval=0.1,  # Faster for testing
            action_delay_min=0.1,
            action_delay_max=0.2
        )
        self.bot = PokerBot(self.config)
    
    def test_bot_initialization(self):
        """Test that the bot initializes correctly"""
        assert self.bot is not None
        assert self.bot.config == self.config
        assert self.bot.cv_engine is not None
        assert self.bot.is_running is False
        assert self.bot.game_context is None
        assert self.bot.hands_played == 0
        assert self.bot.hands_won == 0
        assert self.bot.total_winnings == 0
        assert len(self.bot.actions_history) == 0
    
    def test_bot_with_default_config(self):
        """Test bot with default configuration"""
        default_bot = PokerBot()
        assert default_bot.config is not None
        assert isinstance(default_bot.config, BotConfiguration)
    
    def test_update_game_context_preflop(self):
        """Test updating game context for pre-flop"""
        # Create detection with only player cards (pre-flop)
        player_cards = [
            CardDetection('A', 'h', 0.9, (10, 400, 80, 120), (50, 460)),
            CardDetection('K', 's', 0.85, (100, 400, 80, 120), (140, 460))
        ]
        
        detection = GameStateDetection(
            player_cards=player_cards,
            board_cards=[],
            pot_amount=50,
            player_chips=1000,
            current_bet=10,
            action_buttons={}
        )
        
        self.bot._update_game_context(detection)
        
        assert self.bot.game_context is not None
        assert self.bot.game_context.game_state == GameState.PREFLOP
        assert len(self.bot.game_context.player_cards) == 2
        assert len(self.bot.game_context.board_cards) == 0
    
    def test_update_game_context_flop(self):
        """Test updating game context for flop"""
        player_cards = [
            CardDetection('A', 'h', 0.9, (10, 400, 80, 120), (50, 460)),
            CardDetection('K', 's', 0.85, (100, 400, 80, 120), (140, 460))
        ]
        
        board_cards = [
            CardDetection('Q', 'd', 0.8, (250, 200, 80, 120), (290, 260)),
            CardDetection('J', 'c', 0.75, (350, 200, 80, 120), (390, 260)),
            CardDetection('T', 'h', 0.9, (450, 200, 80, 120), (490, 260))
        ]
        
        detection = GameStateDetection(
            player_cards=player_cards,
            board_cards=board_cards,
            pot_amount=150,
            player_chips=950,
            current_bet=25,
            action_buttons={}
        )
        
        self.bot._update_game_context(detection)
        
        assert self.bot.game_context.game_state == GameState.FLOP
        assert len(self.bot.game_context.player_cards) == 2
        assert len(self.bot.game_context.board_cards) == 3
    
    def test_update_game_context_turn(self):
        """Test updating game context for turn"""
        player_cards = [
            CardDetection('A', 'h', 0.9, (10, 400, 80, 120), (50, 460)),
            CardDetection('K', 's', 0.85, (100, 400, 80, 120), (140, 460))
        ]
        
        board_cards = [
            CardDetection('Q', 'd', 0.8, (250, 200, 80, 120), (290, 260)),
            CardDetection('J', 'c', 0.75, (350, 200, 80, 120), (390, 260)),
            CardDetection('T', 'h', 0.9, (450, 200, 80, 120), (490, 260)),
            CardDetection('9', 's', 0.85, (550, 200, 80, 120), (590, 260))
        ]
        
        detection = GameStateDetection(
            player_cards=player_cards,
            board_cards=board_cards,
            pot_amount=200,
            player_chips=900,
            current_bet=50,
            action_buttons={}
        )
        
        self.bot._update_game_context(detection)
        
        assert self.bot.game_context.game_state == GameState.TURN
        assert len(self.bot.game_context.board_cards) == 4
    
    def test_update_game_context_river(self):
        """Test updating game context for river"""
        player_cards = [
            CardDetection('A', 'h', 0.9, (10, 400, 80, 120), (50, 460)),
            CardDetection('K', 's', 0.85, (100, 400, 80, 120), (140, 460))
        ]
        
        board_cards = [
            CardDetection('Q', 'd', 0.8, (250, 200, 80, 120), (290, 260)),
            CardDetection('J', 'c', 0.75, (350, 200, 80, 120), (390, 260)),
            CardDetection('T', 'h', 0.9, (450, 200, 80, 120), (490, 260)),
            CardDetection('9', 's', 0.85, (550, 200, 80, 120), (590, 260)),
            CardDetection('8', 'd', 0.8, (650, 200, 80, 120), (690, 260))
        ]
        
        detection = GameStateDetection(
            player_cards=player_cards,
            board_cards=board_cards,
            pot_amount=300,
            player_chips=850,
            current_bet=100,
            action_buttons={}
        )
        
        self.bot._update_game_context(detection)
        
        assert self.bot.game_context.game_state == GameState.RIVER
        assert len(self.bot.game_context.board_cards) == 5
    
    def test_should_take_action_validation(self):
        """Test validation logic for taking actions"""
        # No game context - should not take action
        assert self.bot._should_take_action() is False
        
        # Set up valid game context
        player_cards = [
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.KING, Suit.SPADES)
        ]
        
        self.bot.game_context = GameContext(
            game_state=GameState.PREFLOP,
            player_cards=player_cards,
            board_cards=[],
            pot_size=50,
            player_chips=1000,
            current_bet=10,
            position=PlayerPosition.DEALER,
            num_players=6,
            actions_taken=[]
        )
        
        # Valid context - should take action
        assert self.bot._should_take_action() is True
        
        # Invalid context (no cards) - should not take action
        self.bot.game_context.player_cards = []
        assert self.bot._should_take_action() is False
    
    @patch('AutomatedGameplay.CalculateHandRank')
    @patch('AutomatedGameplay.CalculateHandRankValue')
    def test_make_decision_with_strong_hand(self, mock_hand_value, mock_hand_rank):
        """Test decision making with a strong hand"""
        # Mock strong hand
        mock_hand_rank.return_value = 8  # Strong hand rank
        mock_hand_value.return_value = 8000  # Strong hand value
        
        # Set up game context
        player_cards = [
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.ACE, Suit.SPADES)
        ]
        
        self.bot.game_context = GameContext(
            game_state=GameState.PREFLOP,
            player_cards=player_cards,
            board_cards=[],
            pot_size=50,
            player_chips=1000,
            current_bet=10,
            position=PlayerPosition.DEALER,
            num_players=6,
            actions_taken=[]
        )
        
        decision = self.bot._make_decision()
        
        # Should make an aggressive decision with strong hand
        assert decision in [PlayerAction.CALL, PlayerAction.RAISE]
        mock_hand_rank.assert_called_once()
        mock_hand_value.assert_called_once()
    
    @patch('AutomatedGameplay.CalculateHandRank')
    @patch('AutomatedGameplay.CalculateHandRankValue')
    def test_make_decision_with_weak_hand(self, mock_hand_value, mock_hand_rank):
        """Test decision making with a weak hand"""
        # Mock weak hand
        mock_hand_rank.return_value = 1  # Weak hand rank
        mock_hand_value.return_value = 1000  # Weak hand value
        
        player_cards = [
            Card(Rank.DEUCE, Suit.HEARTS),
            Card(Rank.THREE, Suit.SPADES)
        ]
        
        self.bot.game_context = GameContext(
            game_state=GameState.PREFLOP,
            player_cards=player_cards,
            board_cards=[],
            pot_size=50,
            player_chips=1000,
            current_bet=10,
            position=PlayerPosition.DEALER,
            num_players=6,
            actions_taken=[]
        )
        
        decision = self.bot._make_decision()
        
        # Should fold with weak hand
        assert decision == PlayerAction.FOLD
    
    @patch('AutomatedGameplay.CalculateHandRank')
    @patch('AutomatedGameplay.CalculateHandRankValue')
    def test_make_decision_error_handling(self, mock_hand_value, mock_hand_rank):
        """Test error handling in decision making"""
        # Mock exception in hand calculation
        mock_hand_rank.side_effect = Exception("Calculation error")
        
        player_cards = [
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.KING, Suit.SPADES)
        ]
        
        self.bot.game_context = GameContext(
            game_state=GameState.PREFLOP,
            player_cards=player_cards,
            board_cards=[],
            pot_size=50,
            player_chips=1000,
            current_bet=10,
            position=PlayerPosition.DEALER,
            num_players=6,
            actions_taken=[]
        )
        
        decision = self.bot._make_decision()
        
        # Should default to fold on error
        assert decision == PlayerAction.FOLD
    
    def test_decide_action_based_on_strength_preflop(self):
        """Test action decision based on hand strength for pre-flop"""
        self.bot.game_context = GameContext(
            game_state=GameState.PREFLOP,
            player_cards=[],  # Not used in this test
            board_cards=[],
            pot_size=50,
            player_chips=1000,
            current_bet=10,
            position=PlayerPosition.DEALER,
            num_players=6,
            actions_taken=[]
        )
        
        # Very strong hand
        action = self.bot._decide_action_based_on_strength(8, 9000)
        assert action in [PlayerAction.CALL, PlayerAction.RAISE]
        
        # Weak hand
        action = self.bot._decide_action_based_on_strength(1, 1000)
        assert action == PlayerAction.FOLD
        
        # Medium hand
        action = self.bot._decide_action_based_on_strength(4, 6000)
        assert action in [PlayerAction.CALL, PlayerAction.FOLD]
    
    def test_decide_action_based_on_strength_postflop(self):
        """Test action decision based on hand strength for post-flop"""
        self.bot.game_context = GameContext(
            game_state=GameState.FLOP,
            player_cards=[],
            board_cards=[],
            pot_size=150,
            player_chips=950,
            current_bet=25,
            position=PlayerPosition.DEALER,
            num_players=6,
            actions_taken=[]
        )
        
        # Very strong hand
        action = self.bot._decide_action_based_on_strength(8, 8000)
        assert action in [PlayerAction.CALL, PlayerAction.RAISE]
        
        # Weak hand
        action = self.bot._decide_action_based_on_strength(1, 1000)
        assert action == PlayerAction.FOLD
    
    @pytest.mark.asyncio
    @patch('pyautogui.click')
    @patch('AutomatedGameplay.PokerBot._move_mouse_humanlike')
    async def test_click_button(self, mock_move_mouse, mock_click):
        """Test button clicking functionality"""
        mock_move_mouse.return_value = None
        
        button_coords = (100, 200, 120, 40)  # x, y, width, height
        
        await self.bot._click_button(button_coords)
        
        # Should move mouse and click
        mock_move_mouse.assert_called_once()
        mock_click.assert_called_once()
        
        # Check that click coordinates are within button bounds
        click_args = mock_click.call_args[0]
        click_x, click_y = click_args
        assert 100 <= click_x <= 220  # x + width
        assert 200 <= click_y <= 240  # y + height
    
    @pytest.mark.asyncio
    @patch('pyautogui.moveTo')
    @patch('pyautogui.position')
    async def test_move_mouse_humanlike(self, mock_position, mock_move_to):
        """Test human-like mouse movement"""
        mock_position.return_value = (50, 50)  # Current position
        
        target_x, target_y = 200, 300
        
        await self.bot._move_mouse_humanlike(target_x, target_y)
        
        # Should call moveTo with target coordinates
        mock_move_to.assert_called_once()
        args, kwargs = mock_move_to.call_args
        assert args[0] == target_x
        assert args[1] == target_y
        assert 'duration' in kwargs
        assert kwargs['duration'] > 0
    
    @pytest.mark.asyncio
    async def test_execute_action_fold(self):
        """Test executing fold action"""
        action_buttons = {
            'fold': (50, 500, 100, 40),
            'call': (200, 500, 100, 40),
            'raise': (350, 500, 100, 40)
        }
        
        detection = GameStateDetection(
            player_cards=[],
            board_cards=[],
            pot_amount=100,
            player_chips=900,
            current_bet=50,
            action_buttons=action_buttons
        )
        
        with patch.object(self.bot, '_click_button', new_callable=AsyncMock) as mock_click:
            await self.bot._execute_action(PlayerAction.FOLD, detection)
            
            mock_click.assert_called_once_with(action_buttons['fold'])
            assert PlayerAction.FOLD in self.bot.actions_history
    
    @pytest.mark.asyncio
    async def test_execute_action_call(self):
        """Test executing call action"""
        action_buttons = {
            'fold': (50, 500, 100, 40),
            'call': (200, 500, 100, 40),
            'raise': (350, 500, 100, 40)
        }
        
        detection = GameStateDetection(
            player_cards=[],
            board_cards=[],
            pot_amount=100,
            player_chips=900,
            current_bet=50,
            action_buttons=action_buttons
        )
        
        with patch.object(self.bot, '_click_button', new_callable=AsyncMock) as mock_click:
            await self.bot._execute_action(PlayerAction.CALL, detection)
            
            mock_click.assert_called_once_with(action_buttons['call'])
            assert PlayerAction.CALL in self.bot.actions_history
    
    @pytest.mark.asyncio
    async def test_execute_action_missing_button(self):
        """Test executing action when button is missing"""
        action_buttons = {
            'fold': (50, 500, 100, 40),
            # Missing call button
        }
        
        detection = GameStateDetection(
            player_cards=[],
            board_cards=[],
            pot_amount=100,
            player_chips=900,
            current_bet=50,
            action_buttons=action_buttons
        )
        
        with patch.object(self.bot, '_click_button', new_callable=AsyncMock) as mock_click:
            await self.bot._execute_action(PlayerAction.CALL, detection)
            
            # Should not click anything
            mock_click.assert_not_called()
            assert PlayerAction.CALL not in self.bot.actions_history
    
    def test_print_statistics(self):
        """Test statistics printing"""
        # Add some test data
        self.bot.hands_played = 100
        self.bot.hands_won = 30
        self.bot.total_winnings = 500
        self.bot.actions_history = [
            PlayerAction.FOLD, PlayerAction.CALL, PlayerAction.RAISE,
            PlayerAction.FOLD, PlayerAction.CALL, PlayerAction.FOLD
        ]
        
        # This should not crash
        try:
            self.bot._print_statistics()
        except Exception as e:
            pytest.fail(f"Statistics printing failed: {e}")

class TestPokerBotTrainer:
    """Test the poker bot trainer functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.bot = PokerBot()
        self.trainer = PokerBotTrainer(self.bot)
    
    def test_trainer_initialization(self):
        """Test trainer initialization"""
        assert self.trainer.bot == self.bot
        assert isinstance(self.trainer.training_data, list)
        assert isinstance(self.trainer.performance_metrics, list)
    
    def test_analyze_performance_insufficient_data(self):
        """Test performance analysis with insufficient data"""
        # Should handle empty actions gracefully
        try:
            self.trainer._analyze_performance()
        except Exception as e:
            pytest.fail(f"Performance analysis failed with insufficient data: {e}")
    
    def test_analyze_performance_with_data(self):
        """Test performance analysis with sufficient data"""
        # Add test actions
        self.bot.actions_history = [PlayerAction.FOLD] * 40 + [PlayerAction.CALL] * 10
        
        try:
            self.trainer._analyze_performance()
        except Exception as e:
            pytest.fail(f"Performance analysis failed: {e}")
    
    def test_training_session(self):
        """Test training session execution"""
        initial_winnings = self.bot.total_winnings
        
        try:
            self.trainer.start_training_session(num_hands=10)
        except Exception as e:
            pytest.fail(f"Training session failed: {e}")

class TestBotFactoryFunctions:
    """Test bot factory functions"""
    
    def test_create_conservative_bot(self):
        """Test creation of conservative bot"""
        bot = create_conservative_bot()
        
        assert isinstance(bot, PokerBot)
        assert bot.config.aggression_level == 0.3
        assert bot.config.bluff_frequency == 0.05
        assert bot.config.tight_play is True
        assert bot.config.bankroll_management is True
    
    def test_create_aggressive_bot(self):
        """Test creation of aggressive bot"""
        bot = create_aggressive_bot()
        
        assert isinstance(bot, PokerBot)
        assert bot.config.aggression_level == 0.8
        assert bot.config.bluff_frequency == 0.2
        assert bot.config.tight_play is False
        assert bot.config.bankroll_management is True
    
    def test_create_balanced_bot(self):
        """Test creation of balanced bot"""
        bot = create_balanced_bot()
        
        assert isinstance(bot, PokerBot)
        assert bot.config.aggression_level == 0.5
        assert bot.config.bluff_frequency == 0.1
        assert bot.config.tight_play is True
        assert bot.config.bankroll_management is True

@pytest.mark.integration
class TestBotIntegration:
    """Integration tests for the complete bot system"""
    
    def setup_method(self):
        """Set up test environment"""
        self.bot = PokerBot(BotConfiguration(screenshot_interval=0.1))
    
    @patch('AutomatedGameplay.ComputerVisionEngine')
    def test_bot_cv_integration(self, mock_cv_engine):
        """Test integration between bot and computer vision"""
        # Mock CV engine
        mock_instance = Mock()
        mock_cv_engine.return_value = mock_instance
        
        # Create a new bot to use the mocked CV engine
        bot = PokerBot()
        
        assert bot.cv_engine is not None
        mock_cv_engine.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('AutomatedGameplay.ComputerVisionEngine.capture_screen')
    @patch('AutomatedGameplay.ComputerVisionEngine.detect_game_state')
    async def test_main_bot_loop_iteration(self, mock_detect_game_state, mock_capture_screen):
        """Test single iteration of main bot loop"""
        # Mock screen capture
        import numpy as np
        mock_capture_screen.return_value = np.ones((600, 800, 3), dtype=np.uint8)
        
        # Mock game state detection
        mock_detection = GameStateDetection(
            player_cards=[
                CardDetection('A', 'h', 0.9, (10, 400, 80, 120), (50, 460)),
                CardDetection('K', 's', 0.85, (100, 400, 80, 120), (140, 460))
            ],
            board_cards=[],
            pot_amount=50,
            player_chips=1000,
            current_bet=10,
            action_buttons={'fold': (50, 500, 100, 40)}
        )
        mock_detect_game_state.return_value = mock_detection
        
        # Start bot and let it run one iteration
        self.bot.is_running = True
        
        # Patch execute_action to avoid actual clicking
        with patch.object(self.bot, '_execute_action', new_callable=AsyncMock):
            # Run one iteration with timeout
            try:
                await asyncio.wait_for(self.bot._main_bot_loop(), timeout=1.0)
            except asyncio.TimeoutError:
                pass  # Expected - the loop runs indefinitely
        
        # Verify CV methods were called
        mock_capture_screen.assert_called()
        mock_detect_game_state.assert_called()

@pytest.mark.performance
class TestBotPerformance:
    """Performance tests for bot functionality"""
    
    def test_decision_making_speed(self):
        """Test speed of decision making"""
        import time
        
        bot = PokerBot()
        
        # Set up game context
        player_cards = [
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.KING, Suit.SPADES)
        ]
        
        bot.game_context = GameContext(
            game_state=GameState.PREFLOP,
            player_cards=player_cards,
            board_cards=[],
            pot_size=50,
            player_chips=1000,
            current_bet=10,
            position=PlayerPosition.DEALER,
            num_players=6,
            actions_taken=[]
        )
        
        # Time decision making
        start_time = time.time()
        for _ in range(100):
            bot._make_decision()
        end_time = time.time()
        
        avg_decision_time = (end_time - start_time) / 100
        
        # Should be fast (less than 0.01 seconds per decision)
        assert avg_decision_time < 0.01, f"Decision making took {avg_decision_time:.4f} seconds, too slow"

if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])