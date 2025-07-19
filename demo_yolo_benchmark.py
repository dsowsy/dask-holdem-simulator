#!/usr/bin/env python3
"""
Interactive demonstration of YOLO-based poker detection and benchmarking system.
Shows real-time detection, performance comparison, and visualization capabilities.
"""

import sys
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import logging
from pathlib import Path

# Import our YOLO and benchmarking systems
try:
    from YOLODetection import YOLOPokerDetector, PokerObjectType, create_poker_yolo_detector
    from DetectionBenchmark import DetectionBenchmark, run_quick_benchmark
    from ComputerVision import ComputerVisionEngine
except ImportError as e:
    print(f"⚠️  Import error: {e}")
    print("Please ensure all required modules are available.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLOPokerDemo:
    """
    Interactive demonstration of YOLO poker detection capabilities.
    Showcases real-time detection, benchmarking, and visualization.
    """
    
    def __init__(self):
        """Initialize the demo system"""
        print("🎰 YOLO Poker Detection Demo")
        print("=" * 50)
        
        # Initialize detection systems
        self.yolo_detector = None
        self.cv_detector = None
        self.benchmark_system = None
        
        # Demo configuration
        self.demo_scenes = []
        self.current_scene = 0
        
        self._initialize_systems()
        self._create_demo_scenes()
        
        print("✅ Demo system initialized successfully!")
    
    def _initialize_systems(self):
        """Initialize YOLO and comparison systems"""
        print("\n🔧 Initializing detection systems...")
        
        try:
            # Initialize YOLO detector
            print("  - Loading YOLO detector...")
            self.yolo_detector = create_poker_yolo_detector(use_gpu=False)  # Use CPU for demo stability
            print("    ✓ YOLO detector ready")
            
            # Initialize template matching detector for comparison
            print("  - Loading template matching detector...")
            self.cv_detector = ComputerVisionEngine()
            print("    ✓ Template matching detector ready")
            
            # Initialize benchmark system
            print("  - Setting up benchmark system...")
            self.benchmark_system = DetectionBenchmark(
                enable_yolo=True,
                enable_template_matching=True,
                gpu_available=False
            )
            print("    ✓ Benchmark system ready")
            
        except Exception as e:
            logger.error(f"Failed to initialize systems: {e}")
            print(f"⚠️  Falling back to mock systems for demo purposes")
            self._initialize_mock_systems()
    
    def _initialize_mock_systems(self):
        """Initialize mock systems for demonstration"""
        self.yolo_detector = create_poker_yolo_detector(use_gpu=False)
        self.yolo_detector.use_mock = True
        
        self.cv_detector = ComputerVisionEngine()
        
        self.benchmark_system = DetectionBenchmark(
            enable_yolo=True,
            enable_template_matching=True,
            gpu_available=False
        )
    
    def _create_demo_scenes(self):
        """Create various demo poker scenes"""
        print("\n🎨 Creating demo poker scenes...")
        
        # Scene 1: Simple Texas Hold'em hand
        scene1 = self._create_holdem_scene(800, 600, "simple")
        self.demo_scenes.append(("Simple Hold'em Hand", scene1))
        
        # Scene 2: Complex multi-player table
        scene2 = self._create_holdem_scene(1024, 768, "complex")
        self.demo_scenes.append(("Multi-player Table", scene2))
        
        # Scene 3: High-resolution tournament scene
        scene3 = self._create_holdem_scene(1280, 720, "tournament")
        self.demo_scenes.append(("Tournament Scene", scene3))
        
        # Scene 4: Action-heavy scene with chips and buttons
        scene4 = self._create_holdem_scene(800, 600, "action")
        self.demo_scenes.append(("Action Scene", scene4))
        
        print(f"    ✓ Created {len(self.demo_scenes)} demo scenes")
    
    def _create_holdem_scene(self, width: int, height: int, scene_type: str) -> np.ndarray:
        """Create a Texas Hold'em poker scene"""
        # Create poker table background
        image = np.ones((height, width, 3), dtype=np.uint8) * 20
        
        # Draw poker table (green felt)
        center = (width // 2, height // 2)
        table_size = (width // 3, height // 4)
        cv2.ellipse(image, center, table_size, 0, 0, 360, (0, 100, 0), -1)
        cv2.ellipse(image, center, table_size, 0, 0, 360, (255, 255, 255), 3)
        
        if scene_type == "simple":
            self._add_simple_elements(image, width, height)
        elif scene_type == "complex":
            self._add_complex_elements(image, width, height)
        elif scene_type == "tournament":
            self._add_tournament_elements(image, width, height)
        elif scene_type == "action":
            self._add_action_elements(image, width, height)
        
        # Add realistic noise and lighting
        noise = np.random.normal(0, 8, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return image
    
    def _add_simple_elements(self, image: np.ndarray, width: int, height: int):
        """Add elements for simple scene"""
        # Player hole cards
        self._draw_playing_card(image, width//2 - 100, height - 140, 70, 100, "A", "♠")
        self._draw_playing_card(image, width//2 - 20, height - 140, 70, 100, "K", "♠")
        
        # Community cards (flop, turn, river)
        board_x = width//2 - 175
        board_y = height//2 - 50
        community_cards = [("Q", "♠"), ("J", "♠"), ("10", "♠"), ("9", "♥"), ("8", "♦")]
        
        for i, (rank, suit) in enumerate(community_cards):
            if i < 3:  # Flop
                self._draw_playing_card(image, board_x + i * 75, board_y, 65, 95, rank, suit)
            else:  # Turn and River
                self._draw_playing_card(image, board_x + i * 75 + 20, board_y, 65, 95, rank, suit)
        
        # Simple chip stack
        self._draw_chip_stack(image, width//2 - 150, height//2 + 80, 35, 500, (128, 0, 128))
    
    def _add_complex_elements(self, image: np.ndarray, width: int, height: int):
        """Add elements for complex multi-player scene"""
        # Multiple player positions (6-player table)
        player_positions = [
            (50, height//2 - 50),      # Left
            (width - 150, height//2 - 50),  # Right
            (width//4, 50),            # Top left
            (3*width//4, 50),          # Top right
            (width//4, height - 150),  # Bottom left
            (3*width//4, height - 150) # Bottom right
        ]
        
        for i, (x, y) in enumerate(player_positions):
            # Player hole cards (face down for opponents)
            if i == 0:  # Hero position - face up
                self._draw_playing_card(image, x, y, 50, 75, "A", "♥")
                self._draw_playing_card(image, x + 55, y, 50, 75, "A", "♦")
            else:  # Opponents - face down
                self._draw_card_back(image, x, y, 50, 75)
                self._draw_card_back(image, x + 55, y, 50, 75)
            
            # Player chip stacks
            if x + 120 < width and y + 80 < height:
                stack_value = 100 * (i + 1)
                color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)][i]
                self._draw_chip_stack(image, x + 60, y + 80, 25, stack_value, color)
        
        # Community cards
        self._add_simple_elements(image, width, height)
        
        # Dealer button
        self._draw_dealer_button(image, width//2 + 120, height//2 + 120)
        
        # Pot
        self._draw_chip_stack(image, width//2, height//2 + 120, 40, 1500, (255, 255, 0))
    
    def _add_tournament_elements(self, image: np.ndarray, width: int, height: int):
        """Add elements for tournament scene"""
        self._add_complex_elements(image, width, height)
        
        # Tournament info overlay
        cv2.rectangle(image, (10, 10), (300, 100), (0, 0, 0), -1)
        cv2.rectangle(image, (10, 10), (300, 100), (255, 255, 255), 2)
        cv2.putText(image, "TOURNAMENT", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, "Blinds: 100/200", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, "Level 5", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, "Players: 847", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _add_action_elements(self, image: np.ndarray, width: int, height: int):
        """Add elements for action scene with buttons"""
        self._add_simple_elements(image, width, height)
        
        # Action buttons
        button_y = height - 50
        buttons = [
            ("FOLD", (50, button_y), (100, 100, 100)),
            ("CALL 200", (160, button_y), (0, 200, 0)),
            ("RAISE", (300, button_y), (200, 0, 0)),
            ("ALL-IN", (440, button_y), (255, 215, 0))
        ]
        
        for text, (x, y), color in buttons:
            cv2.rectangle(image, (x, y), (x + 100, y + 35), color, -1)
            cv2.rectangle(image, (x, y), (x + 100, y + 35), (255, 255, 255), 2)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_x = x + (100 - text_size[0]) // 2
            cv2.putText(image, text, (text_x, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    def _draw_playing_card(self, image: np.ndarray, x: int, y: int, w: int, h: int, rank: str, suit: str):
        """Draw a playing card with rank and suit"""
        # Card background
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), -1)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 2)
        
        # Card color (red for hearts/diamonds, black for clubs/spades)
        color = (0, 0, 255) if suit in "♥♦" else (0, 0, 0)
        
        # Draw rank
        rank_scale = 0.6 if len(rank) == 1 else 0.5
        cv2.putText(image, rank, (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, rank_scale, color, 2)
        
        # Draw suit
        cv2.putText(image, suit, (x + 5, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Small rank/suit in bottom right (upside down)
        cv2.putText(image, rank, (x + w - 15, y + h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        cv2.putText(image, suit, (x + w - 15, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def _draw_card_back(self, image: np.ndarray, x: int, y: int, w: int, h: int):
        """Draw a face-down card"""
        # Card background
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 150), -1)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)
        
        # Pattern
        for i in range(5, w, 10):
            for j in range(5, h, 10):
                cv2.circle(image, (x + i, y + j), 2, (100, 100, 200), -1)
    
    def _draw_chip_stack(self, image: np.ndarray, x: int, y: int, radius: int, value: int, color: tuple):
        """Draw a stack of poker chips"""
        # Calculate stack height
        stack_height = min(max(value // 100, 1), 8)
        
        # Draw chips in stack
        for i in range(stack_height):
            offset = (stack_height - i - 1) * 3
            cv2.circle(image, (x, y - offset), radius, color, -1)
            cv2.circle(image, (x, y - offset), radius, (255, 255, 255), 2)
        
        # Value label
        if value >= 1000:
            text = f"{value//1000}K"
        else:
            text = str(value)
        
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        cv2.putText(image, text, (x - text_size[0]//2, y + radius + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _draw_dealer_button(self, image: np.ndarray, x: int, y: int):
        """Draw dealer button"""
        cv2.circle(image, (x, y), 20, (255, 215, 0), -1)
        cv2.circle(image, (x, y), 20, (255, 255, 255), 2)
        cv2.putText(image, "D", (x - 8, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    def run_interactive_demo(self):
        """Run interactive demonstration"""
        print("\n🎮 Starting Interactive Demo")
        print("=" * 40)
        print("Commands:")
        print("  [SPACE] - Next scene")
        print("  [b] - Run benchmark")
        print("  [d] - Toggle detection visualization")
        print("  [s] - Save current detection results")
        print("  [q] - Quit demo")
        print("=" * 40)
        
        show_detections = True
        
        while True:
            try:
                # Get current scene
                scene_name, scene_image = self.demo_scenes[self.current_scene]
                
                print(f"\n📸 Current Scene: {scene_name}")
                
                # Run YOLO detection
                start_time = time.time()
                yolo_result = self.yolo_detector.detect_poker_objects(scene_image)
                yolo_time = time.time() - start_time
                
                # Create visualization
                if show_detections:
                    vis_image = self.yolo_detector.visualize_detections(scene_image, yolo_result)
                else:
                    vis_image = scene_image.copy()
                
                # Add demo info overlay
                self._add_demo_overlay(vis_image, yolo_result, yolo_time, scene_name, show_detections)
                
                # Display image
                cv2.imshow('YOLO Poker Detection Demo', vis_image)
                
                # Print detection summary
                self._print_detection_summary(yolo_result, yolo_time)
                
                # Handle user input
                key = cv2.waitKey(0) & 0xFF
                
                if key == ord('q'):
                    print("👋 Exiting demo...")
                    break
                elif key == ord(' '):
                    self.current_scene = (self.current_scene + 1) % len(self.demo_scenes)
                elif key == ord('b'):
                    self._run_benchmark_demo()
                elif key == ord('d'):
                    show_detections = not show_detections
                    print(f"Detection visualization: {'ON' if show_detections else 'OFF'}")
                elif key == ord('s'):
                    self._save_detection_results(scene_name, yolo_result)
                
            except KeyboardInterrupt:
                print("\n👋 Demo interrupted by user")
                break
            except Exception as e:
                logger.error(f"Demo error: {e}")
                print(f"⚠️  Error in demo: {e}")
                continue
        
        cv2.destroyAllWindows()
        print("✅ Demo completed")
    
    def _add_demo_overlay(self, image: np.ndarray, result, detection_time: float, scene_name: str, show_detections: bool):
        """Add demo information overlay to image"""
        # Info panel background
        panel_height = 120
        cv2.rectangle(image, (10, 10), (400, panel_height), (0, 0, 0), -1)
        cv2.rectangle(image, (10, 10), (400, panel_height), (255, 255, 255), 2)
        
        # Demo info
        y_pos = 30
        cv2.putText(image, f"Scene: {scene_name}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_pos += 20
        cv2.putText(image, f"Detection Time: {detection_time*1000:.1f}ms", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_pos += 15
        fps = 1.0 / detection_time if detection_time > 0 else 0
        cv2.putText(image, f"FPS: {fps:.1f}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_pos += 15
        total_objects = len(result.cards) + len(result.chips) + len(result.players) + len(result.markers)
        cv2.putText(image, f"Objects Found: {total_objects}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_pos += 15
        cv2.putText(image, f"Detections: {'ON' if show_detections else 'OFF'}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Controls hint
        cv2.putText(image, "Press 'h' for help", (image.shape[1] - 200, image.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _print_detection_summary(self, result, detection_time: float):
        """Print detection summary to console"""
        print(f"  ⏱️  Detection Time: {detection_time*1000:.2f}ms ({1.0/detection_time:.1f} FPS)")
        print(f"  🃏 Cards: {len(result.cards)}")
        print(f"  🔴 Chips: {len(result.chips)}")
        print(f"  👤 Players: {len(result.players)}")
        print(f"  📍 Markers: {len(result.markers)}")
        
        if result.cards:
            print("  Card details:")
            for i, card in enumerate(result.cards[:5]):  # Show first 5 cards
                rank = card.card_rank or "?"
                suit = card.card_suit or "?"
                conf = card.confidence
                print(f"    {i+1}. {rank}{suit} (confidence: {conf:.2f})")
    
    def _run_benchmark_demo(self):
        """Run benchmark demonstration"""
        print("\n🏁 Running Performance Benchmark...")
        print("This may take a moment...")
        
        try:
            # Create test images for benchmarking
            test_images = []
            for scene_name, scene_image in self.demo_scenes:
                test_images.append(scene_image)
            
            # Add some additional varied images
            for i in range(6):  # Total 10 images
                extra_image = self.benchmark_system.create_test_images(1)[0]
                test_images.append(extra_image)
            
            # Run speed benchmark
            result = self.benchmark_system.run_speed_benchmark(test_images)
            
            # Display results
            report = self.benchmark_system.generate_performance_report()
            print(report)
            
            # Save results
            self.benchmark_system.save_results("demo_benchmark_results.json")
            print("💾 Benchmark results saved to 'demo_benchmark_results.json'")
            
            # Create plots if matplotlib available
            try:
                self.benchmark_system.create_performance_plots("demo_plots")
                print("📊 Performance plots saved to 'demo_plots/' directory")
            except Exception as e:
                print(f"⚠️  Could not create plots: {e}")
            
        except Exception as e:
            logger.error(f"Benchmark demo failed: {e}")
            print(f"⚠️  Benchmark demo failed: {e}")
        
        print("\nPress any key to continue demo...")
        cv2.waitKey(0)
    
    def _save_detection_results(self, scene_name: str, result):
        """Save detection results for current scene"""
        filename = f"detection_results_{scene_name.lower().replace(' ', '_')}.json"
        self.yolo_detector.save_detection_results(result, filename)
        print(f"💾 Detection results saved to '{filename}'")
    
    def run_quick_comparison(self):
        """Run a quick comparison between YOLO and template matching"""
        print("\n⚡ Quick Performance Comparison")
        print("=" * 40)
        
        # Use first demo scene
        scene_name, test_image = self.demo_scenes[0]
        
        print(f"Testing with: {scene_name}")
        print("Running multiple detections for average timing...")
        
        # YOLO timing
        yolo_times = []
        for i in range(5):
            start_time = time.time()
            yolo_result = self.yolo_detector.detect_poker_objects(test_image)
            yolo_times.append(time.time() - start_time)
        
        # Template matching timing
        cv_times = []
        for i in range(5):
            start_time = time.time()
            cv_result = self.cv_detector.detect_game_state(test_image)
            cv_times.append(time.time() - start_time)
        
        # Results
        yolo_avg = np.mean(yolo_times) * 1000
        cv_avg = np.mean(cv_times) * 1000
        
        print(f"\n📊 Results:")
        print(f"YOLO Detection:       {yolo_avg:.2f}ms avg ({1000/yolo_avg:.1f} FPS)")
        print(f"Template Matching:    {cv_avg:.2f}ms avg ({1000/cv_avg:.1f} FPS)")
        print(f"Speed Advantage:      {cv_avg/yolo_avg:.1f}x {'faster' if yolo_avg < cv_avg else 'slower'}")
        
        print(f"\n🎯 Object Detection:")
        print(f"YOLO Objects:         {len(yolo_result.cards) + len(yolo_result.chips) + len(yolo_result.players)}")
        print(f"Template Matching:    {len(cv_result.player_cards) + len(cv_result.board_cards)}")

def main():
    """Main demo function"""
    print("🎰 YOLO Poker Detection & Benchmarking Demo")
    print("=" * 50)
    
    try:
        # Create demo system
        demo = YOLOPokerDemo()
        
        # Show menu
        while True:
            print("\n🎮 Demo Options:")
            print("1. Interactive Visual Demo")
            print("2. Quick Performance Comparison")
            print("3. Full Benchmark Suite")
            print("4. Exit")
            
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == "1":
                demo.run_interactive_demo()
            elif choice == "2":
                demo.run_quick_comparison()
            elif choice == "3":
                report = run_quick_benchmark(num_images=15)
                print(report)
            elif choice == "4":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please select 1-4.")
                
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"💥 Demo failed: {e}")
        print("This might be due to missing dependencies or system configuration.")
        print("The demo can still run in mock mode for testing purposes.")

if __name__ == "__main__":
    main()