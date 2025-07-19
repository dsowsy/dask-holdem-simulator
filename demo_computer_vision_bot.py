#!/usr/bin/env python3
"""
Demonstration script for Computer Vision and Automated Gameplay system.
Shows the complete pipeline from image analysis to automated poker decisions.
"""

import numpy as np
import cv2
import asyncio
import time
from typing import List
import logging

from ComputerVision import ComputerVisionEngine, CardDetection, GameStateDetection
from AutomatedGameplay import (
    PokerBot, 
    BotConfiguration, 
    create_conservative_bot,
    create_aggressive_bot,
    create_balanced_bot
)
from HandCalculations import CalculateHandRank, CalculateHandRankValue
from utils import PlayerAction

# Configure logging for demo
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PokerGameDemo:
    """Demonstration class for the poker computer vision and automation system"""
    
    def __init__(self):
        self.cv_engine = ComputerVisionEngine()
        logger.info("Initialized Computer Vision Engine")
        
    def create_demo_screenshot(self, scenario: str = "strong_hand") -> np.ndarray:
        """Create different demo scenarios for testing"""
        # Create a 800x600 poker table image
        screenshot = np.ones((600, 800, 3), dtype=np.uint8) * 30  # Dark background
        
        # Draw poker table
        cv2.ellipse(screenshot, (400, 300), (350, 200), 0, 0, 360, (0, 80, 0), -1)
        cv2.ellipse(screenshot, (400, 300), (350, 200), 0, 0, 360, (255, 255, 255), 3)
        
        # Define scenarios
        scenarios = {
            "strong_hand": {
                "player_cards": [("A", "h"), ("A", "s")],
                "board_cards": [("A", "d"), ("K", "c"), ("Q", "h")],
                "pot": 250,
                "chips": 1500,
                "bet": 50
            },
            "weak_hand": {
                "player_cards": [("2", "h"), ("7", "c")],
                "board_cards": [("K", "s"), ("Q", "d"), ("J", "h")],
                "pot": 100,
                "chips": 800,
                "bet": 25
            },
            "straight_draw": {
                "player_cards": [("9", "h"), ("T", "s")],
                "board_cards": [("J", "d"), ("Q", "c"), ("2", "h")],
                "pot": 150,
                "chips": 1200,
                "bet": 30
            },
            "flush_draw": {
                "player_cards": [("A", "h"), ("K", "h")],
                "board_cards": [("Q", "h"), ("J", "h"), ("3", "s")],
                "pot": 200,
                "chips": 1000,
                "bet": 40
            }
        }
        
        scenario_data = scenarios.get(scenario, scenarios["strong_hand"])
        
        # Draw player cards
        player_positions = [(150, 450), (250, 450)]
        for i, ((rank, suit), (x, y)) in enumerate(zip(scenario_data["player_cards"], player_positions)):
            self._draw_card(screenshot, x, y, rank, suit)
        
        # Draw board cards
        board_positions = [(250, 200), (350, 200), (450, 200), (550, 200), (650, 200)]
        for i, ((rank, suit), (x, y)) in enumerate(zip(scenario_data["board_cards"], board_positions)):
            if i < len(scenario_data["board_cards"]):
                self._draw_card(screenshot, x, y, rank, suit)
        
        # Draw action buttons
        self._draw_action_buttons(screenshot)
        
        # Draw game information
        self._draw_game_info(screenshot, scenario_data["pot"], scenario_data["chips"], scenario_data["bet"])
        
        return screenshot
    
    def _draw_card(self, img: np.ndarray, x: int, y: int, rank: str, suit: str):
        """Draw a playing card on the image"""
        # Card background
        cv2.rectangle(img, (x, y), (x + 80, y + 120), (255, 255, 255), -1)
        cv2.rectangle(img, (x, y), (x + 80, y + 120), (0, 0, 0), 2)
        
        # Suit colors
        suit_colors = {
            'h': (0, 0, 255),    # Hearts - Red
            'd': (0, 0, 255),    # Diamonds - Red  
            'c': (0, 0, 0),      # Clubs - Black
            's': (0, 0, 0)       # Spades - Black
        }
        
        # Draw rank
        cv2.putText(img, rank, (x + 15, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        
        # Draw suit
        suit_color = suit_colors.get(suit, (0, 0, 0))
        cv2.putText(img, suit, (x + 15, y + 70), cv2.FONT_HERSHEY_SIMPLEX, 1, suit_color, 2)
        
        # Add corner indicators (smaller)
        cv2.putText(img, rank, (x + 50, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(img, suit, (x + 50, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, suit_color, 1)
    
    def _draw_action_buttons(self, img: np.ndarray):
        """Draw action buttons on the image"""
        button_y = 520
        buttons = [
            ('FOLD', (50, button_y, 100, 40), (100, 100, 100)),
            ('CALL', (200, button_y, 100, 40), (100, 200, 100)),
            ('RAISE', (350, button_y, 100, 40), (200, 100, 100))
        ]
        
        for text, (x, y, w, h), color in buttons:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
            
            # Center text
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = x + (w - text_size[0]) // 2
            text_y = y + (h + text_size[1]) // 2
            cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def _draw_game_info(self, img: np.ndarray, pot: int, chips: int, bet: int):
        """Draw game information on the image"""
        # Pot information (center)
        pot_text = f"POT: ${pot}"
        cv2.putText(img, pot_text, (320, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Player chips (bottom left)
        chips_text = f"CHIPS: ${chips}"
        cv2.putText(img, chips_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Current bet (top right)
        bet_text = f"BET: ${bet}"
        cv2.putText(img, bet_text, (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Game state indicator
        cv2.putText(img, "FLOP", (350, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    def demonstrate_computer_vision(self, scenario: str = "strong_hand"):
        """Demonstrate computer vision capabilities"""
        logger.info(f"=== Computer Vision Demo - {scenario.upper()} ===")
        
        # Create demo screenshot
        screenshot = self.create_demo_screenshot(scenario)
        
        # Detect game state
        start_time = time.time()
        game_state = self.cv_engine.detect_game_state(screenshot)
        detection_time = time.time() - start_time
        
        logger.info(f"Detection completed in {detection_time:.3f} seconds")
        
        # Report detected cards
        logger.info(f"Detected {len(game_state.player_cards)} player cards:")
        for i, card in enumerate(game_state.player_cards):
            logger.info(f"  Card {i+1}: {card.rank}{card.suit} (confidence: {card.confidence:.2f})")
        
        logger.info(f"Detected {len(game_state.board_cards)} board cards:")
        for i, card in enumerate(game_state.board_cards):
            logger.info(f"  Card {i+1}: {card.rank}{card.suit} (confidence: {card.confidence:.2f})")
        
        # Convert to poker objects and calculate hand strength
        player_cards = self.cv_engine.cards_to_poker_objects(game_state.player_cards)
        board_cards = self.cv_engine.cards_to_poker_objects(game_state.board_cards)
        
        if len(player_cards) >= 2:
            try:
                hand_rank = CalculateHandRank(player_cards, board_cards)
                hand_value = CalculateHandRankValue(player_cards, board_cards)
                logger.info(f"Hand strength - Rank: {hand_rank}, Value: {hand_value}")
            except Exception as e:
                logger.warning(f"Could not calculate hand strength: {e}")
        
        # Save debug image
        debug_filename = f"demo_detection_{scenario}.jpg"
        self.cv_engine.save_detection_debug_image(screenshot, 
                                                game_state.player_cards + game_state.board_cards,
                                                debug_filename)
        logger.info(f"Debug image saved as {debug_filename}")
        
        return game_state, screenshot
    
    def demonstrate_bot_decision_making(self, scenario: str = "strong_hand"):
        """Demonstrate automated decision making"""
        logger.info(f"=== Bot Decision Making Demo - {scenario.upper()} ===")
        
        # Get game state from computer vision
        game_state, screenshot = self.demonstrate_computer_vision(scenario)
        
        # Test different bot configurations
        bots = {
            "Conservative": create_conservative_bot(),
            "Aggressive": create_aggressive_bot(),
            "Balanced": create_balanced_bot()
        }
        
        for bot_name, bot in bots.items():
            logger.info(f"\n--- {bot_name} Bot Analysis ---")
            
            # Update bot's game context
            bot._update_game_context(game_state)
            
            if bot.game_context and bot._should_take_action():
                decision = bot._make_decision()
                logger.info(f"{bot_name} bot decision: {decision}")
                
                # Show reasoning
                if hasattr(bot, 'config'):
                    logger.info(f"  Aggression level: {bot.config.aggression_level}")
                    logger.info(f"  Tight play: {bot.config.tight_play}")
                    logger.info(f"  Bluff frequency: {bot.config.bluff_frequency}")
            else:
                logger.info(f"{bot_name} bot: No action taken (insufficient information)")
    
    async def demonstrate_automated_gameplay(self, scenario: str = "strong_hand"):
        """Demonstrate complete automated gameplay"""
        logger.info(f"=== Automated Gameplay Demo - {scenario.upper()} ===")
        
        # Create bot with demo configuration
        bot_config = BotConfiguration(
            aggression_level=0.6,
            action_delay_min=0.5,
            action_delay_max=1.0,
            screenshot_interval=0.2
        )
        bot = PokerBot(bot_config)
        
        # Get game state
        game_state, screenshot = self.demonstrate_computer_vision(scenario)
        
        # Update bot context
        bot._update_game_context(game_state)
        
        if bot.game_context and bot._should_take_action():
            logger.info("Bot is ready to take action")
            
            # Make decision
            decision = bot._make_decision()
            logger.info(f"Bot decision: {decision}")
            
            # Simulate action execution (without actually clicking)
            logger.info("Simulating action execution...")
            await asyncio.sleep(1.0)  # Simulate human-like delay
            
            logger.info(f"Action {decision} would be executed")
            bot.actions_history.append(decision)
            
            # Show statistics
            bot._print_statistics()
        else:
            logger.info("Bot cannot take action in current state")
    
    def run_comprehensive_demo(self):
        """Run a comprehensive demonstration of all system capabilities"""
        logger.info("=" * 60)
        logger.info("POKER COMPUTER VISION & AUTOMATION SYSTEM DEMO")
        logger.info("=" * 60)
        
        scenarios = ["strong_hand", "weak_hand", "straight_draw", "flush_draw"]
        
        for scenario in scenarios:
            logger.info("\n" + "=" * 50)
            logger.info(f"SCENARIO: {scenario.upper().replace('_', ' ')}")
            logger.info("=" * 50)
            
            # Demonstrate computer vision
            self.demonstrate_computer_vision(scenario)
            
            # Demonstrate bot decision making
            self.demonstrate_bot_decision_making(scenario)
            
            # Brief pause between scenarios
            time.sleep(1)
        
        logger.info("\n" + "=" * 60)
        logger.info("DEMO COMPLETED")
        logger.info("=" * 60)
    
    async def run_live_demo(self):
        """Run a live demo showing continuous operation"""
        logger.info("=== LIVE DEMO - Continuous Operation ===")
        
        bot = create_balanced_bot()
        scenarios = ["strong_hand", "weak_hand", "straight_draw", "flush_draw"]
        
        for i in range(10):  # Simulate 10 hands
            scenario = scenarios[i % len(scenarios)]
            logger.info(f"\n--- Hand {i+1}: {scenario.replace('_', ' ').title()} ---")
            
            # Get game state
            game_state, _ = self.demonstrate_computer_vision(scenario)
            
            # Update bot
            bot._update_game_context(game_state)
            
            if bot._should_take_action():
                decision = bot._make_decision()
                logger.info(f"Hand {i+1} - Bot action: {decision}")
                bot.actions_history.append(decision)
                
                # Simulate hand outcome (random for demo)
                import random
                if random.random() > 0.5:
                    bot.hands_won += 1
                    winnings = random.randint(50, 200)
                    bot.total_winnings += winnings
                    logger.info(f"Hand {i+1} - Won ${winnings}")
                else:
                    loss = random.randint(20, 100)
                    bot.total_winnings -= loss
                    logger.info(f"Hand {i+1} - Lost ${loss}")
                
                bot.hands_played += 1
            
            await asyncio.sleep(0.5)  # Brief pause between hands
        
        # Final statistics
        logger.info("\n=== FINAL STATISTICS ===")
        bot._print_statistics()

def main():
    """Main demonstration function"""
    demo = PokerGameDemo()
    
    print("Poker Computer Vision & Automation Demo")
    print("Choose demo mode:")
    print("1. Comprehensive Demo (all scenarios)")
    print("2. Single Scenario Demo")
    print("3. Live Demo (continuous operation)")
    print("4. Computer Vision Only")
    print("5. Bot Decision Making Only")
    
    try:
        choice = input("Enter choice (1-5): ").strip()
        
        if choice == "1":
            demo.run_comprehensive_demo()
        
        elif choice == "2":
            print("\nAvailable scenarios:")
            print("- strong_hand (pair of aces)")
            print("- weak_hand (2-7 offsuit)")
            print("- straight_draw (9-T)")
            print("- flush_draw (A-K suited)")
            
            scenario = input("Enter scenario: ").strip()
            if scenario not in ["strong_hand", "weak_hand", "straight_draw", "flush_draw"]:
                scenario = "strong_hand"
            
            demo.demonstrate_computer_vision(scenario)
            demo.demonstrate_bot_decision_making(scenario)
        
        elif choice == "3":
            print("Running live demo...")
            asyncio.run(demo.run_live_demo())
        
        elif choice == "4":
            scenario = input("Enter scenario (or press Enter for strong_hand): ").strip() or "strong_hand"
            demo.demonstrate_computer_vision(scenario)
        
        elif choice == "5":
            scenario = input("Enter scenario (or press Enter for strong_hand): ").strip() or "strong_hand"
            demo.demonstrate_bot_decision_making(scenario)
        
        else:
            print("Invalid choice. Running comprehensive demo...")
            demo.run_comprehensive_demo()
    
    except KeyboardInterrupt:
        logger.info("\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo error: {e}")

if __name__ == "__main__":
    main()