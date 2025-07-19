"""
Automated Gameplay module for poker bot automation.
Integrates computer vision with automated mouse/keyboard actions.
"""

import pyautogui
import time
import random
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import threading
import asyncio
from pynput import mouse, keyboard

from ComputerVision import ComputerVisionEngine, GameStateDetection, CardDetection
from Decision import PokerDecision, DecisionType
from HandCalculations import CalculateHandRank, CalculateHandRankValue
from utils import PlayerAction, PlayerPosition
from poker import Card

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GameState(Enum):
    """Current state of the poker game"""
    WAITING_FOR_GAME = "waiting_for_game"
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"
    HAND_ENDED = "hand_ended"

@dataclass
class BotConfiguration:
    """Configuration for the poker bot"""
    aggression_level: float = 0.5  # 0.0 = very passive, 1.0 = very aggressive
    bluff_frequency: float = 0.1   # Frequency of bluffing
    tight_play: bool = True        # Whether to play tight or loose
    position_awareness: bool = True # Whether to consider position in decisions
    bankroll_management: bool = True # Whether to manage bankroll
    max_buy_in_percentage: float = 0.05  # Max percentage of bankroll for buy-in
    
    # Computer vision settings
    screenshot_interval: float = 0.5  # How often to take screenshots
    confidence_threshold: float = 0.7  # Minimum confidence for card detection
    
    # Automation settings
    action_delay_min: float = 1.0  # Minimum delay before taking action
    action_delay_max: float = 3.0  # Maximum delay before taking action
    mouse_movement_speed: float = 0.3  # Speed of mouse movements

@dataclass
class GameContext:
    """Current context of the poker game"""
    game_state: GameState
    player_cards: List[Card]
    board_cards: List[Card]
    pot_size: int
    player_chips: int
    current_bet: int
    position: PlayerPosition
    num_players: int
    actions_taken: List[PlayerAction]

class PokerBot:
    """Main poker bot class that combines computer vision and automated gameplay"""
    
    def __init__(self, config: BotConfiguration = None):
        self.config = config or BotConfiguration()
        self.cv_engine = ComputerVisionEngine()
        self.is_running = False
        self.game_context = None
        self.decision_engine = None
        
        # Statistics tracking
        self.hands_played = 0
        self.hands_won = 0
        self.total_winnings = 0
        self.actions_history = []
        
        # Safety settings
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1
        
    def start_bot(self, target_window_title: str = None):
        """Start the automated poker bot"""
        logger.info("Starting poker bot...")
        self.is_running = True
        
        # Focus on the poker application window if specified
        if target_window_title:
            self._focus_window(target_window_title)
        
        # Start the main bot loop
        try:
            asyncio.run(self._main_bot_loop())
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Bot encountered an error: {e}")
        finally:
            self.stop_bot()
    
    def stop_bot(self):
        """Stop the automated poker bot"""
        logger.info("Stopping poker bot...")
        self.is_running = False
        self._print_statistics()
    
    async def _main_bot_loop(self):
        """Main loop for the poker bot"""
        while self.is_running:
            try:
                # Capture current game state
                screenshot = self.cv_engine.capture_screen()
                if screenshot.size == 0:
                    await asyncio.sleep(self.config.screenshot_interval)
                    continue
                
                # Detect game state using computer vision
                game_state_detection = self.cv_engine.detect_game_state(screenshot)
                
                # Update game context
                self._update_game_context(game_state_detection)
                
                # Make decision based on current state
                if self.game_context and self._should_take_action():
                    decision = self._make_decision()
                    if decision:
                        await self._execute_action(decision, game_state_detection)
                
                # Wait before next iteration
                await asyncio.sleep(self.config.screenshot_interval)
                
            except Exception as e:
                logger.error(f"Error in main bot loop: {e}")
                await asyncio.sleep(1.0)
    
    def _focus_window(self, window_title: str):
        """Focus on the specified window"""
        try:
            # This is a simplified implementation
            # In practice, you'd use platform-specific window management
            logger.info(f"Attempting to focus window: {window_title}")
            # pyautogui.getWindowsWithTitle() could be used on Windows
        except Exception as e:
            logger.warning(f"Could not focus window {window_title}: {e}")
    
    def _update_game_context(self, detection: GameStateDetection):
        """Update the current game context based on detection results"""
        if not detection.player_cards and not detection.board_cards:
            return
        
        # Convert detected cards to poker objects
        player_cards = self.cv_engine.cards_to_poker_objects(detection.player_cards)
        board_cards = self.cv_engine.cards_to_poker_objects(detection.board_cards)
        
        # Determine game state based on number of board cards
        if len(board_cards) == 0:
            game_state = GameState.PREFLOP
        elif len(board_cards) == 3:
            game_state = GameState.FLOP
        elif len(board_cards) == 4:
            game_state = GameState.TURN
        elif len(board_cards) == 5:
            game_state = GameState.RIVER
        else:
            game_state = GameState.WAITING_FOR_GAME
        
        # Update context
        old_context = self.game_context
        self.game_context = GameContext(
            game_state=game_state,
            player_cards=player_cards,
            board_cards=board_cards,
            pot_size=detection.pot_amount or 0,
            player_chips=detection.player_chips or 0,
            current_bet=detection.current_bet or 0,
            position=PlayerPosition.NONE,  # Would need additional detection
            num_players=6,  # Would need additional detection
            actions_taken=[]
        )
        
        # Log state changes
        if old_context is None or old_context.game_state != self.game_context.game_state:
            logger.info(f"Game state changed to: {self.game_context.game_state}")
            logger.info(f"Player cards: {[str(card) for card in player_cards]}")
            logger.info(f"Board cards: {[str(card) for card in board_cards]}")
    
    def _should_take_action(self) -> bool:
        """Determine if the bot should take an action"""
        if not self.game_context:
            return False
        
        # Check if we have valid cards
        if len(self.game_context.player_cards) < 2:
            return False
        
        # Add logic to detect if it's our turn
        # This would typically involve detecting highlighted action buttons
        # or other UI indicators
        return True
    
    def _make_decision(self) -> Optional[PlayerAction]:
        """Make a decision based on current game state and cards"""
        if not self.game_context or len(self.game_context.player_cards) < 2:
            return None
        
        try:
            # Calculate hand strength
            hand_rank = CalculateHandRank(
                self.game_context.player_cards,
                self.game_context.board_cards
            )
            
            hand_value = CalculateHandRankValue(
                self.game_context.player_cards,
                self.game_context.board_cards
            )
            
            # Simple decision logic based on hand strength
            decision = self._decide_action_based_on_strength(hand_rank, hand_value)
            
            logger.info(f"Decision made: {decision} (hand_rank: {hand_rank}, hand_value: {hand_value})")
            return decision
            
        except Exception as e:
            logger.error(f"Error making decision: {e}")
            return PlayerAction.FOLD  # Safe default
    
    def _decide_action_based_on_strength(self, hand_rank: int, hand_value: float) -> PlayerAction:
        """Decide action based on hand strength and bot configuration"""
        # Normalize hand value (this is a simplified approach)
        normalized_strength = min(1.0, max(0.0, hand_value / 10000.0))
        
        # Adjust for bot configuration
        aggression_modifier = self.config.aggression_level
        
        # Pre-flop strategy
        if self.game_context.game_state == GameState.PREFLOP:
            if normalized_strength > 0.8:
                return PlayerAction.RAISE if random.random() < aggression_modifier else PlayerAction.CALL
            elif normalized_strength > 0.6:
                return PlayerAction.CALL
            elif normalized_strength > 0.4 and not self.config.tight_play:
                return PlayerAction.CALL
            else:
                return PlayerAction.FOLD
        
        # Post-flop strategy
        else:
            if normalized_strength > 0.7:
                return PlayerAction.RAISE if random.random() < aggression_modifier else PlayerAction.CALL
            elif normalized_strength > 0.5:
                return PlayerAction.CALL
            elif normalized_strength > 0.3 and random.random() < self.config.bluff_frequency:
                return PlayerAction.RAISE  # Bluff
            else:
                return PlayerAction.FOLD
    
    async def _execute_action(self, action: PlayerAction, detection: GameStateDetection):
        """Execute the decided action through mouse/keyboard automation"""
        try:
            # Add human-like delay
            delay = random.uniform(
                self.config.action_delay_min,
                self.config.action_delay_max
            )
            await asyncio.sleep(delay)
            
            # Get action button coordinates
            action_buttons = detection.action_buttons
            
            if action == PlayerAction.FOLD and 'fold' in action_buttons:
                await self._click_button(action_buttons['fold'])
            elif action == PlayerAction.CALL and 'call' in action_buttons:
                await self._click_button(action_buttons['call'])
            elif action == PlayerAction.CHECK and 'check' in action_buttons:
                await self._click_button(action_buttons['check'])
            elif action == PlayerAction.RAISE and 'raise' in action_buttons:
                await self._click_button(action_buttons['raise'])
            elif action == PlayerAction.BET and 'bet' in action_buttons:
                await self._click_button(action_buttons['bet'])
            else:
                logger.warning(f"Could not execute action {action}: button not found")
                return
            
            # Log the action
            self.actions_history.append(action)
            logger.info(f"Executed action: {action}")
            
        except Exception as e:
            logger.error(f"Error executing action {action}: {e}")
    
    async def _click_button(self, button_coords: Tuple[int, int, int, int]):
        """Click a button with human-like mouse movement"""
        x, y, w, h = button_coords
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Add some randomness to click position
        click_x = center_x + random.randint(-w//4, w//4)
        click_y = center_y + random.randint(-h//4, h//4)
        
        # Move mouse with human-like curve
        await self._move_mouse_humanlike(click_x, click_y)
        
        # Click with small delay
        pyautogui.click(click_x, click_y)
        await asyncio.sleep(0.1)
    
    async def _move_mouse_humanlike(self, target_x: int, target_y: int):
        """Move mouse to target position with human-like movement"""
        current_x, current_y = pyautogui.position()
        
        # Calculate distance and duration
        distance = ((target_x - current_x) ** 2 + (target_y - current_y) ** 2) ** 0.5
        duration = distance * self.config.mouse_movement_speed / 1000
        
        # Move with smooth movement
        pyautogui.moveTo(target_x, target_y, duration=duration, tween=pyautogui.easeInOutQuad)
    
    def _print_statistics(self):
        """Print bot performance statistics"""
        logger.info("=== Bot Statistics ===")
        logger.info(f"Hands played: {self.hands_played}")
        logger.info(f"Hands won: {self.hands_won}")
        logger.info(f"Win rate: {self.hands_won / max(1, self.hands_played) * 100:.1f}%")
        logger.info(f"Total winnings: {self.total_winnings}")
        logger.info(f"Actions taken: {len(self.actions_history)}")
        
        # Action frequency analysis
        if self.actions_history:
            action_counts = {}
            for action in self.actions_history:
                action_counts[action] = action_counts.get(action, 0) + 1
            
            logger.info("Action frequency:")
            for action, count in action_counts.items():
                percentage = count / len(self.actions_history) * 100
                logger.info(f"  {action}: {count} ({percentage:.1f}%)")

class PokerBotTrainer:
    """Training module for the poker bot using reinforcement learning concepts"""
    
    def __init__(self, bot: PokerBot):
        self.bot = bot
        self.training_data = []
        self.performance_metrics = []
    
    def start_training_session(self, num_hands: int = 100):
        """Start a training session to improve bot performance"""
        logger.info(f"Starting training session for {num_hands} hands")
        
        initial_winnings = self.bot.total_winnings
        
        # Run the bot for the specified number of hands
        # This would be integrated with the main bot loop
        
        final_winnings = self.bot.total_winnings
        session_profit = final_winnings - initial_winnings
        
        logger.info(f"Training session completed. Profit: {session_profit}")
        
        # Analyze performance and adjust strategy
        self._analyze_performance()
    
    def _analyze_performance(self):
        """Analyze bot performance and suggest improvements"""
        if len(self.bot.actions_history) < 10:
            return
        
        # Simple performance analysis
        recent_actions = self.bot.actions_history[-50:]
        fold_rate = recent_actions.count(PlayerAction.FOLD) / len(recent_actions)
        
        logger.info(f"Recent fold rate: {fold_rate * 100:.1f}%")
        
        # Suggest adjustments
        if fold_rate > 0.8:
            logger.info("Suggestion: Bot is playing too tight, consider loosening strategy")
        elif fold_rate < 0.3:
            logger.info("Suggestion: Bot is playing too loose, consider tightening strategy")

# Utility functions for bot management
def create_conservative_bot() -> PokerBot:
    """Create a conservative poker bot configuration"""
    config = BotConfiguration(
        aggression_level=0.3,
        bluff_frequency=0.05,
        tight_play=True,
        bankroll_management=True
    )
    return PokerBot(config)

def create_aggressive_bot() -> PokerBot:
    """Create an aggressive poker bot configuration"""
    config = BotConfiguration(
        aggression_level=0.8,
        bluff_frequency=0.2,
        tight_play=False,
        bankroll_management=True
    )
    return PokerBot(config)

def create_balanced_bot() -> PokerBot:
    """Create a balanced poker bot configuration"""
    config = BotConfiguration(
        aggression_level=0.5,
        bluff_frequency=0.1,
        tight_play=True,
        bankroll_management=True
    )
    return PokerBot(config)