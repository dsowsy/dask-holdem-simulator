"""
Comprehensive benchmarking system for comparing poker detection approaches.
Compares YOLO vs Template Matching performance, accuracy, and speed.
"""

import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import json
import logging
from pathlib import Path
import statistics
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import gc

# Import our detection systems
from YOLODetection import YOLOPokerDetector, PokerGameDetection as YOLOGameDetection
from ComputerVision import ComputerVisionEngine, GameStateDetection as CVGameDetection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DetectionMethod(Enum):
    """Detection methods to benchmark"""
    YOLO = "yolo"
    TEMPLATE_MATCHING = "template_matching"
    CNN_BASED = "cnn_based"

@dataclass
class BenchmarkMetrics:
    """Metrics for detection performance"""
    method: DetectionMethod
    detection_time: float
    fps: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float]
    
    # Accuracy metrics
    cards_detected: int
    chips_detected: int
    players_detected: int
    markers_detected: int
    total_objects_detected: int
    
    # Precision/Recall (when ground truth available)
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    
    # Object-specific accuracy
    card_accuracy: Optional[float] = None
    chip_accuracy: Optional[float] = None
    
    # Additional metrics
    false_positives: int = 0
    false_negatives: int = 0
    confidence_avg: float = 0.0
    confidence_std: float = 0.0

@dataclass
class BenchmarkResult:
    """Complete benchmark result"""
    test_name: str
    image_count: int
    total_time: float
    metrics_by_method: Dict[DetectionMethod, List[BenchmarkMetrics]]
    summary_stats: Dict[str, Any]
    
class PerformanceProfiler:
    """System performance profiler for benchmarking"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_cpu = 0
        self.start_memory = 0
        self.start_time = 0
        
    def start(self):
        """Start performance monitoring"""
        self.start_cpu = self.process.cpu_percent()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.start_time = time.time()
        
    def stop(self) -> Tuple[float, float, float]:
        """Stop monitoring and return metrics"""
        end_time = time.time()
        cpu_usage = self.process.cpu_percent()
        memory_usage = self.process.memory_info().rss / 1024 / 1024  # MB
        
        elapsed_time = end_time - self.start_time
        avg_cpu = (self.start_cpu + cpu_usage) / 2
        memory_delta = memory_usage - self.start_memory
        
        return elapsed_time, avg_cpu, memory_delta

class DetectionBenchmark:
    """
    Comprehensive benchmarking system for poker object detection.
    Compares different detection methods across various metrics.
    """
    
    def __init__(self, 
                 enable_yolo: bool = True,
                 enable_template_matching: bool = True,
                 enable_cnn: bool = True,
                 gpu_available: bool = None):
        """
        Initialize benchmark system.
        
        Args:
            enable_yolo: Whether to benchmark YOLO detection
            enable_template_matching: Whether to benchmark template matching
            enable_cnn: Whether to benchmark CNN-based detection
            gpu_available: Whether GPU is available (auto-detect if None)
        """
        self.enable_yolo = enable_yolo
        self.enable_template_matching = enable_template_matching
        self.enable_cnn = enable_cnn
        
        # Initialize detectors
        self.detectors = {}
        self._initialize_detectors(gpu_available)
        
        # Benchmark configuration
        self.warmup_frames = 5
        self.test_frames = 50
        self.profiler = PerformanceProfiler()
        
        # Results storage
        self.results = []
        
    def _initialize_detectors(self, gpu_available: Optional[bool]):
        """Initialize detection systems"""
        if gpu_available is None:
            gpu_available = self._check_gpu_availability()
        
        try:
            if self.enable_yolo:
                self.detectors[DetectionMethod.YOLO] = YOLOPokerDetector(use_gpu=gpu_available)
                logger.info("✓ YOLO detector initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize YOLO: {e}")
            self.enable_yolo = False
            
        try:
            if self.enable_template_matching or self.enable_cnn:
                self.detectors[DetectionMethod.TEMPLATE_MATCHING] = ComputerVisionEngine()
                logger.info("✓ Template matching detector initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize template matching: {e}")
            self.enable_template_matching = False
            self.enable_cnn = False
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available for acceleration"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            try:
                return cv2.cuda.getCudaEnabledDeviceCount() > 0
            except:
                return False
    
    def create_test_images(self, count: int = 50) -> List[np.ndarray]:
        """
        Create synthetic test images for benchmarking.
        
        Args:
            count: Number of test images to create
            
        Returns:
            List of test images
        """
        logger.info(f"Creating {count} synthetic test images...")
        
        images = []
        
        for i in range(count):
            # Create varied test scenarios
            if i % 4 == 0:
                # Simple scenario: few objects
                image = self._create_simple_poker_scene(800, 600)
            elif i % 4 == 1:
                # Complex scenario: many objects
                image = self._create_complex_poker_scene(1024, 768)
            elif i % 4 == 2:
                # High-resolution scenario
                image = self._create_simple_poker_scene(1920, 1080)
            else:
                # Different aspect ratio
                image = self._create_simple_poker_scene(1280, 720)
                
            images.append(image)
            
        logger.info(f"✓ Created {len(images)} test images")
        return images
    
    def _create_simple_poker_scene(self, width: int, height: int) -> np.ndarray:
        """Create a simple poker scene with basic objects"""
        # Create dark poker table background
        image = np.ones((height, width, 3), dtype=np.uint8) * 30
        
        # Draw poker table
        center = (width // 2, height // 2)
        table_size = (min(width, height) // 3, min(width, height) // 4)
        cv2.ellipse(image, center, table_size, 0, 0, 360, (0, 80, 0), -1)
        cv2.ellipse(image, center, table_size, 0, 0, 360, (255, 255, 255), 3)
        
        # Add player cards (2 cards)
        card_positions = [(width//2 - 100, height - 150), (width//2, height - 150)]
        for i, (x, y) in enumerate(card_positions):
            self._draw_card(image, x, y, 80, 120, f"A{'hs'[i]}")
        
        # Add board cards (5 cards)
        board_start_x = width//2 - 200
        board_y = height//2 - 60
        board_cards = ['K', 'Q', 'J', 'T', '9']
        suits = ['h', 'd', 'c', 's', 'h']
        
        for i, (rank, suit) in enumerate(zip(board_cards, suits)):
            x = board_start_x + i * 85
            self._draw_card(image, x, board_y, 80, 120, f"{rank}{suit}")
        
        # Add chips
        chip_positions = [(width//2 - 150, height//2 + 100), (width//2 + 50, height//2 + 100)]
        chip_values = [100, 500]
        
        for (x, y), value in zip(chip_positions, chip_values):
            self._draw_chip_stack(image, x, y, 40, value)
        
        # Add some noise for realism
        noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
        image = cv2.add(image, noise)
        
        return image
    
    def _create_complex_poker_scene(self, width: int, height: int) -> np.ndarray:
        """Create a complex poker scene with many objects"""
        # Start with simple scene
        image = self._create_simple_poker_scene(width, height)
        
        # Add more players (simulate 6-player table)
        player_positions = [
            (50, height//2), (width-150, height//2),  # Left and right
            (width//4, 50), (3*width//4, 50),        # Top
            (width//4, height-150), (3*width//4, height-150)  # Bottom
        ]
        
        for i, (x, y) in enumerate(player_positions):
            # Player cards (face down)
            if x + 80 < width and y + 120 < height:
                self._draw_card(image, x, y, 60, 90, "back", face_down=True)
                if x + 140 < width:
                    self._draw_card(image, x + 65, y, 60, 90, "back", face_down=True)
            
            # Player chips
            if x + 100 < width and y + 150 < height:
                self._draw_chip_stack(image, x, y + 95, 30, 25 * (i + 1))
        
        # Add dealer button
        dealer_x, dealer_y = width//2 + 100, height//2 + 150
        cv2.circle(image, (dealer_x, dealer_y), 15, (255, 255, 0), -1)
        cv2.putText(image, "D", (dealer_x-5, dealer_y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Add action buttons
        button_y = height - 50
        buttons = [("FOLD", (50, button_y)), ("CALL", (150, button_y)), ("RAISE", (250, button_y))]
        
        for text, (x, y) in buttons:
            cv2.rectangle(image, (x, y), (x + 80, y + 30), (100, 100, 100), -1)
            cv2.rectangle(image, (x, y), (x + 80, y + 30), (255, 255, 255), 2)
            cv2.putText(image, text, (x + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return image
    
    def _draw_card(self, image: np.ndarray, x: int, y: int, w: int, h: int, 
                   card_label: str, face_down: bool = False):
        """Draw a playing card on the image"""
        if face_down:
            # Draw card back
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 150), -1)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)
            # Add pattern
            for i in range(5, w, 10):
                for j in range(5, h, 10):
                    cv2.circle(image, (x + i, y + j), 2, (100, 100, 200), -1)
        else:
            # Draw card face
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), -1)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 2)
            
            if len(card_label) >= 2:
                rank, suit = card_label[0], card_label[1]
                
                # Color based on suit
                color = (0, 0, 255) if suit in 'hd' else (0, 0, 0)  # Red for hearts/diamonds
                
                # Draw rank
                cv2.putText(image, rank, (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                # Draw suit symbol
                suit_symbol = {'h': '♥', 'd': '♦', 'c': '♣', 's': '♠'}.get(suit, suit)
                cv2.putText(image, suit_symbol, (x + 5, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def _draw_chip_stack(self, image: np.ndarray, x: int, y: int, radius: int, value: int):
        """Draw a stack of poker chips"""
        # Choose color based on value
        colors = {
            1: (255, 255, 255),    # White
            5: (0, 0, 255),        # Red  
            25: (0, 255, 0),       # Green
            100: (0, 0, 0),        # Black
            500: (128, 0, 128),    # Purple
            1000: (255, 255, 0),   # Yellow
        }
        
        # Find closest color
        color = colors.get(value, (128, 128, 128))
        for chip_value in sorted(colors.keys()):
            if value >= chip_value:
                color = colors[chip_value]
        
        # Draw stack (multiple circles for 3D effect)
        stack_height = min(value // 25, 8)  # Max 8 chips in visual stack
        
        for i in range(stack_height):
            offset = stack_height - i - 1
            cv2.circle(image, (x, y - offset * 2), radius, color, -1)
            cv2.circle(image, (x, y - offset * 2), radius, (255, 255, 255), 2)
        
        # Add value text
        if value >= 1000:
            text = f"{value//1000}K"
        else:
            text = str(value)
        cv2.putText(image, text, (x - 10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    def run_speed_benchmark(self, test_images: List[np.ndarray]) -> BenchmarkResult:
        """
        Run speed-focused benchmark comparing detection methods.
        
        Args:
            test_images: List of test images
            
        Returns:
            Benchmark results
        """
        logger.info("🚀 Running speed benchmark...")
        
        results_by_method = {}
        
        for method in DetectionMethod:
            if method not in self.detectors:
                continue
                
            logger.info(f"Testing {method.value}...")
            
            detector = self.detectors[method]
            method_results = []
            
            # Warmup
            logger.info(f"  Warming up ({self.warmup_frames} frames)...")
            for i in range(min(self.warmup_frames, len(test_images))):
                self._run_single_detection(detector, test_images[i], method)
            
            # Actual benchmark
            logger.info(f"  Running benchmark ({len(test_images)} frames)...")
            for i, image in enumerate(test_images):
                if i % 10 == 0:
                    logger.info(f"    Frame {i+1}/{len(test_images)}")
                
                metrics = self._run_single_detection(detector, image, method)
                method_results.append(metrics)
                
                # Force garbage collection periodically
                if i % 20 == 0:
                    gc.collect()
            
            results_by_method[method] = method_results
            logger.info(f"  ✓ {method.value} completed")
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_stats(results_by_method)
        
        result = BenchmarkResult(
            test_name="Speed Benchmark",
            image_count=len(test_images),
            total_time=sum(sum(r.detection_time for r in results) 
                         for results in results_by_method.values()),
            metrics_by_method=results_by_method,
            summary_stats=summary_stats
        )
        
        self.results.append(result)
        logger.info("✅ Speed benchmark completed")
        
        return result
    
    def run_accuracy_benchmark(self, test_images: List[np.ndarray], 
                             ground_truth: Optional[List[Dict]] = None) -> BenchmarkResult:
        """
        Run accuracy-focused benchmark.
        
        Args:
            test_images: List of test images
            ground_truth: Optional ground truth annotations
            
        Returns:
            Benchmark results
        """
        logger.info("🎯 Running accuracy benchmark...")
        
        results_by_method = {}
        
        for method in DetectionMethod:
            if method not in self.detectors:
                continue
                
            logger.info(f"Testing {method.value} accuracy...")
            
            detector = self.detectors[method]
            method_results = []
            
            for i, image in enumerate(test_images):
                if i % 10 == 0:
                    logger.info(f"    Frame {i+1}/{len(test_images)}")
                
                metrics = self._run_single_detection(detector, image, method)
                
                # Add accuracy metrics if ground truth available
                if ground_truth and i < len(ground_truth):
                    self._calculate_accuracy_metrics(metrics, ground_truth[i])
                
                method_results.append(metrics)
            
            results_by_method[method] = method_results
            logger.info(f"  ✓ {method.value} accuracy test completed")
        
        summary_stats = self._calculate_summary_stats(results_by_method)
        
        result = BenchmarkResult(
            test_name="Accuracy Benchmark",
            image_count=len(test_images),
            total_time=0,  # Not the focus of this benchmark
            metrics_by_method=results_by_method,
            summary_stats=summary_stats
        )
        
        self.results.append(result)
        logger.info("✅ Accuracy benchmark completed")
        
        return result
    
    def run_scalability_benchmark(self, image_sizes: List[Tuple[int, int]]) -> BenchmarkResult:
        """
        Run scalability benchmark across different image sizes.
        
        Args:
            image_sizes: List of (width, height) tuples to test
            
        Returns:
            Benchmark results
        """
        logger.info("📏 Running scalability benchmark...")
        
        results_by_method = {}
        
        for method in DetectionMethod:
            if method not in self.detectors:
                continue
                
            logger.info(f"Testing {method.value} scalability...")
            
            detector = self.detectors[method]
            method_results = []
            
            for width, height in image_sizes:
                logger.info(f"  Testing resolution: {width}x{height}")
                
                # Create test image at this resolution
                test_image = self._create_simple_poker_scene(width, height)
                
                # Run multiple tests at this resolution
                resolution_times = []
                for _ in range(5):  # 5 runs per resolution
                    metrics = self._run_single_detection(detector, test_image, method)
                    resolution_times.append(metrics.detection_time)
                
                # Create summary metrics for this resolution
                avg_metrics = BenchmarkMetrics(
                    method=method,
                    detection_time=statistics.mean(resolution_times),
                    fps=1.0 / statistics.mean(resolution_times),
                    cpu_usage=0,  # Not tracked in scalability test
                    memory_usage=0,
                    gpu_usage=None,
                    cards_detected=0,  # Not the focus
                    chips_detected=0,
                    players_detected=0,
                    markers_detected=0,
                    total_objects_detected=0
                )
                
                method_results.append(avg_metrics)
            
            results_by_method[method] = method_results
            logger.info(f"  ✓ {method.value} scalability test completed")
        
        summary_stats = self._calculate_summary_stats(results_by_method)
        
        result = BenchmarkResult(
            test_name="Scalability Benchmark",
            image_count=len(image_sizes),
            total_time=0,
            metrics_by_method=results_by_method,
            summary_stats=summary_stats
        )
        
        self.results.append(result)
        logger.info("✅ Scalability benchmark completed")
        
        return result
    
    def _run_single_detection(self, detector, image: np.ndarray, method: DetectionMethod) -> BenchmarkMetrics:
        """Run detection on a single image and collect metrics"""
        
        # Start performance monitoring
        self.profiler.start()
        
        # Run detection based on method
        if method == DetectionMethod.YOLO:
            detection_result = detector.detect_poker_objects(image)
            cards_count = len(detection_result.cards)
            chips_count = len(detection_result.chips)
            players_count = len(detection_result.players)
            markers_count = len(detection_result.markers)
            detection_time = detection_result.detection_time
            
            # Calculate confidence stats
            all_detections = (detection_result.cards + detection_result.chips + 
                            detection_result.players + detection_result.markers)
            if all_detections:
                confidences = [d.confidence for d in all_detections]
                confidence_avg = statistics.mean(confidences)
                confidence_std = statistics.stdev(confidences) if len(confidences) > 1 else 0
            else:
                confidence_avg = confidence_std = 0
            
        else:  # Template matching or CNN
            start_time = time.time()
            cv_result = detector.detect_game_state(image)
            detection_time = time.time() - start_time
            
            cards_count = len(cv_result.player_cards) + len(cv_result.board_cards)
            chips_count = 0  # Not tracked in current CV system
            players_count = 0  # Not tracked in current CV system  
            markers_count = 0  # Not tracked in current CV system
            confidence_avg = confidence_std = 0  # Not available in current system
        
        # Stop performance monitoring
        elapsed_time, cpu_usage, memory_usage = self.profiler.stop()
        
        # Get GPU usage if available
        gpu_usage = self._get_gpu_usage() if method == DetectionMethod.YOLO else None
        
        # Calculate FPS
        fps = 1.0 / detection_time if detection_time > 0 else 0
        
        return BenchmarkMetrics(
            method=method,
            detection_time=detection_time,
            fps=fps,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage,
            cards_detected=cards_count,
            chips_detected=chips_count,
            players_detected=players_count,
            markers_detected=markers_count,
            total_objects_detected=cards_count + chips_count + players_count + markers_count,
            confidence_avg=confidence_avg,
            confidence_std=confidence_std
        )
    
    def _get_gpu_usage(self) -> Optional[float]:
        """Get GPU usage percentage"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].load * 100  # GPU utilization percentage
        except ImportError:
            pass
        return None
    
    def _calculate_accuracy_metrics(self, metrics: BenchmarkMetrics, ground_truth: Dict):
        """Calculate accuracy metrics against ground truth"""
        # This would be implemented with actual ground truth data
        # For now, we'll simulate some accuracy calculations
        pass
    
    def _calculate_summary_stats(self, results_by_method: Dict) -> Dict[str, Any]:
        """Calculate summary statistics across all methods"""
        summary = {}
        
        for method, method_results in results_by_method.items():
            if not method_results:
                continue
                
            detection_times = [r.detection_time for r in method_results]
            fps_values = [r.fps for r in method_results]
            cpu_usage = [r.cpu_usage for r in method_results]
            memory_usage = [r.memory_usage for r in method_results]
            
            summary[method.value] = {
                'avg_detection_time': statistics.mean(detection_times),
                'std_detection_time': statistics.stdev(detection_times) if len(detection_times) > 1 else 0,
                'min_detection_time': min(detection_times),
                'max_detection_time': max(detection_times),
                'avg_fps': statistics.mean(fps_values),
                'max_fps': max(fps_values),
                'avg_cpu_usage': statistics.mean(cpu_usage),
                'avg_memory_usage': statistics.mean(memory_usage),
                'total_objects_detected': sum(r.total_objects_detected for r in method_results)
            }
        
        return summary
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        if not self.results:
            return "No benchmark results available."
        
        report = []
        report.append("=" * 80)
        report.append("🏆 POKER DETECTION BENCHMARK REPORT")
        report.append("=" * 80)
        
        for result in self.results:
            report.append(f"\n📊 {result.test_name}")
            report.append("-" * 60)
            report.append(f"Images tested: {result.image_count}")
            
            # Performance comparison table
            if result.summary_stats:
                report.append("\nPerformance Summary:")
                report.append(f"{'Method':<20} {'Avg Time (ms)':<15} {'FPS':<10} {'CPU %':<10} {'Objects':<10}")
                report.append("-" * 70)
                
                for method, stats in result.summary_stats.items():
                    avg_time_ms = stats['avg_detection_time'] * 1000
                    report.append(f"{method:<20} {avg_time_ms:<15.2f} {stats['avg_fps']:<10.1f} "
                                f"{stats['avg_cpu_usage']:<10.1f} {stats['total_objects_detected']:<10}")
                
                # Winner analysis
                fastest_method = min(result.summary_stats.items(), 
                                   key=lambda x: x[1]['avg_detection_time'])
                highest_fps = max(result.summary_stats.items(), 
                                key=lambda x: x[1]['avg_fps'])
                
                report.append(f"\n🥇 Fastest Detection: {fastest_method[0]} "
                            f"({fastest_method[1]['avg_detection_time']*1000:.2f}ms)")
                report.append(f"🚀 Highest FPS: {highest_fps[0]} "
                            f"({highest_fps[1]['avg_fps']:.1f} FPS)")
        
        # Overall recommendations
        report.append("\n" + "=" * 80)
        report.append("📝 RECOMMENDATIONS")
        report.append("=" * 80)
        
        if DetectionMethod.YOLO in [list(r.metrics_by_method.keys())[0] for r in self.results if r.metrics_by_method]:
            report.append("✅ YOLO Detection:")
            report.append("   - Best for real-time applications requiring high FPS")
            report.append("   - Excellent for detecting multiple object types simultaneously")
            report.append("   - Requires GPU for optimal performance")
            report.append("   - Higher accuracy for complex scenes")
        
        if DetectionMethod.TEMPLATE_MATCHING in [list(r.metrics_by_method.keys())[0] for r in self.results if r.metrics_by_method]:
            report.append("\n✅ Template Matching:")
            report.append("   - Good for CPU-only environments")
            report.append("   - Lower memory requirements")
            report.append("   - More predictable performance")
            report.append("   - Better for simple, controlled environments")
        
        report.append(f"\n🔗 Integration Recommendation:")
        report.append("   - Use YOLO for primary detection with GPU")
        report.append("   - Fallback to template matching when GPU unavailable")
        report.append("   - Hybrid approach: YOLO for object detection + CNN for card recognition")
        
        return "\n".join(report)
    
    def save_results(self, filename: str):
        """Save benchmark results to JSON file"""
        results_data = []
        
        for result in self.results:
            result_dict = {
                'test_name': result.test_name,
                'image_count': result.image_count,
                'total_time': result.total_time,
                'summary_stats': result.summary_stats,
                'detailed_metrics': {}
            }
            
            for method, metrics_list in result.metrics_by_method.items():
                result_dict['detailed_metrics'][method.value] = [
                    asdict(metrics) for metrics in metrics_list
                ]
            
            results_data.append(result_dict)
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"✅ Benchmark results saved to {filename}")
    
    def create_performance_plots(self, save_dir: str = "benchmark_plots"):
        """Create performance visualization plots"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        if not self.results:
            logger.warning("No results to plot")
            return
        
        for result in self.results:
            if not result.summary_stats:
                continue
                
            # FPS comparison plot
            methods = list(result.summary_stats.keys())
            fps_values = [result.summary_stats[method]['avg_fps'] for method in methods]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(methods, fps_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            plt.title(f'{result.test_name} - FPS Comparison')
            plt.ylabel('Frames Per Second (FPS)')
            plt.xlabel('Detection Method')
            
            # Add value labels on bars
            for bar, fps in zip(bars, fps_values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{fps:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{result.test_name.lower().replace(' ', '_')}_fps.png", dpi=300)
            plt.close()
            
            # Detection time comparison plot
            detection_times = [result.summary_stats[method]['avg_detection_time'] * 1000 
                             for method in methods]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(methods, detection_times, color=['#FFE66D', '#FF6B6B', '#A8E6CF'])
            plt.title(f'{result.test_name} - Detection Time Comparison')
            plt.ylabel('Average Detection Time (ms)')
            plt.xlabel('Detection Method')
            
            for bar, time_ms in zip(bars, detection_times):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{time_ms:.1f}ms', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{result.test_name.lower().replace(' ', '_')}_time.png", dpi=300)
            plt.close()
        
        logger.info(f"✅ Performance plots saved to {save_dir}/")

# Convenience function for quick benchmarking
def run_quick_benchmark(num_images: int = 20) -> str:
    """
    Run a quick benchmark comparison between detection methods.
    
    Args:
        num_images: Number of test images to use
        
    Returns:
        Performance report string
    """
    logger.info("🚀 Starting quick benchmark...")
    
    # Create benchmark system
    benchmark = DetectionBenchmark()
    
    # Create test images
    test_images = benchmark.create_test_images(num_images)
    
    # Run speed benchmark
    speed_result = benchmark.run_speed_benchmark(test_images)
    
    # Generate report
    report = benchmark.generate_performance_report()
    
    # Save results
    benchmark.save_results("quick_benchmark_results.json")
    
    return report