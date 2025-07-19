"""
Comprehensive test suite for YOLO-based poker detection and benchmarking system.
Tests YOLO integration, benchmarking accuracy, and performance metrics.
"""

import unittest
import numpy as np
import cv2
import time
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Import modules to test
from YOLODetection import (
    YOLOPokerDetector, PokerObjectType, YOLODetection, 
    PokerGameDetection, create_poker_yolo_detector
)
from DetectionBenchmark import (
    DetectionBenchmark, BenchmarkMetrics, DetectionMethod,
    PerformanceProfiler, run_quick_benchmark
)

class TestYOLODetection(unittest.TestCase):
    """Test YOLO-based poker object detection"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_image = self._create_test_image()
        
    def _create_test_image(self) -> np.ndarray:
        """Create a test poker scene image"""
        # Create a simple test image
        image = np.ones((600, 800, 3), dtype=np.uint8) * 50
        
        # Draw some basic shapes representing poker objects
        # Card-like rectangles
        cv2.rectangle(image, (100, 100), (180, 220), (255, 255, 255), -1)
        cv2.rectangle(image, (200, 100), (280, 220), (255, 255, 255), -1)
        
        # Chip-like circles
        cv2.circle(image, (150, 350), 30, (255, 0, 0), -1)
        cv2.circle(image, (250, 350), 30, (0, 255, 0), -1)
        
        # Player area rectangles
        cv2.rectangle(image, (50, 450), (150, 550), (100, 100, 255), -1)
        cv2.rectangle(image, (650, 450), (750, 550), (100, 100, 255), -1)
        
        return image
    
    @patch('YOLODetection.torch')
    def test_yolo_detector_initialization(self, mock_torch):
        """Test YOLO detector initialization"""
        # Mock torch availability
        mock_torch.cuda.is_available.return_value = True
        mock_torch.hub.load.return_value = Mock()
        
        detector = YOLOPokerDetector(use_gpu=True)
        
        self.assertIsNotNone(detector)
        self.assertEqual(detector.confidence_threshold, 0.5)
        self.assertEqual(detector.nms_threshold, 0.4)
        self.assertEqual(detector.input_size, (640, 640))
    
    def test_yolo_detector_mock_detection(self):
        """Test YOLO detector with mock backend"""
        detector = YOLOPokerDetector(use_gpu=False)
        detector.use_mock = True  # Force mock mode
        
        result = detector.detect_poker_objects(self.test_image)
        
        self.assertIsInstance(result, PokerGameDetection)
        self.assertGreater(len(result.cards), 0)
        self.assertGreater(len(result.chips), 0)
        self.assertGreater(result.detection_time, 0)
        
        # Check card detections
        for card in result.cards:
            self.assertIsInstance(card, YOLODetection)
            self.assertEqual(card.object_type, PokerObjectType.CARD)
            self.assertIsNotNone(card.card_rank)
            self.assertIsNotNone(card.card_suit)
    
    def test_poker_object_type_enum(self):
        """Test PokerObjectType enum"""
        self.assertEqual(PokerObjectType.CARD.value, "card")
        self.assertEqual(PokerObjectType.CHIP.value, "chip")
        self.assertEqual(PokerObjectType.PLAYER.value, "player")
        self.assertEqual(PokerObjectType.DEALER_BUTTON.value, "dealer_button")
    
    def test_yolo_detection_dataclass(self):
        """Test YOLODetection dataclass"""
        detection = YOLODetection(
            object_type=PokerObjectType.CARD,
            confidence=0.95,
            bbox=(100, 100, 80, 120),
            center=(140, 160),
            class_id=0,
            label="card_ah",
            card_rank="A",
            card_suit="h"
        )
        
        self.assertEqual(detection.object_type, PokerObjectType.CARD)
        self.assertEqual(detection.confidence, 0.95)
        self.assertEqual(detection.bbox, (100, 100, 80, 120))
        self.assertEqual(detection.card_rank, "A")
        self.assertEqual(detection.card_suit, "h")
    
    def test_poker_game_detection_dataclass(self):
        """Test PokerGameDetection dataclass"""
        card_detection = YOLODetection(
            object_type=PokerObjectType.CARD,
            confidence=0.9,
            bbox=(0, 0, 80, 120),
            center=(40, 60),
            class_id=0,
            label="card_ah"
        )
        
        game_detection = PokerGameDetection(
            cards=[card_detection],
            chips=[],
            players=[],
            markers=[],
            action_buttons=[],
            detection_time=0.05,
            frame_size=(800, 600)
        )
        
        self.assertEqual(len(game_detection.cards), 1)
        self.assertEqual(game_detection.detection_time, 0.05)
        self.assertEqual(game_detection.frame_size, (800, 600))
    
    def test_performance_stats(self):
        """Test performance statistics tracking"""
        detector = YOLOPokerDetector(use_gpu=False)
        detector.use_mock = True
        
        # Run multiple detections
        for _ in range(5):
            detector.detect_poker_objects(self.test_image)
        
        stats = detector.get_performance_stats()
        
        self.assertIn('average_detection_time', stats)
        self.assertIn('fps', stats)
        self.assertIn('total_frames', stats)
        self.assertEqual(stats['total_frames'], 5)
        self.assertGreater(stats['fps'], 0)
    
    def test_visualization(self):
        """Test detection visualization"""
        detector = YOLOPokerDetector(use_gpu=False)
        detector.use_mock = True
        
        detections = detector.detect_poker_objects(self.test_image)
        vis_image = detector.visualize_detections(self.test_image, detections)
        
        self.assertEqual(vis_image.shape, self.test_image.shape)
        self.assertFalse(np.array_equal(vis_image, self.test_image))  # Should be different
    
    def test_save_detection_results(self):
        """Test saving detection results to JSON"""
        detector = YOLOPokerDetector(use_gpu=False)
        detector.use_mock = True
        
        detections = detector.detect_poker_objects(self.test_image)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            detector.save_detection_results(detections, f.name)
            
            # Read back and verify
            with open(f.name, 'r') as read_file:
                data = json.load(read_file)
                
                self.assertIn('detection_time', data)
                self.assertIn('cards', data)
                self.assertIn('chips', data)
                self.assertGreater(len(data['cards']), 0)

class TestDetectionBenchmark(unittest.TestCase):
    """Test benchmarking system for detection methods"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_images = self._create_test_images(5)
        
    def _create_test_images(self, count: int) -> List[np.ndarray]:
        """Create test images for benchmarking"""
        images = []
        for i in range(count):
            # Create varied test images
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            # Add some structure
            cv2.rectangle(image, (100 + i*20, 100), (200 + i*20, 200), (255, 255, 255), -1)
            images.append(image)
        return images
    
    @patch('DetectionBenchmark.YOLOPokerDetector')
    @patch('DetectionBenchmark.ComputerVisionEngine')
    def test_benchmark_initialization(self, mock_cv, mock_yolo):
        """Test benchmark system initialization"""
        benchmark = DetectionBenchmark()
        
        self.assertIsInstance(benchmark, DetectionBenchmark)
        self.assertTrue(benchmark.enable_yolo)
        self.assertTrue(benchmark.enable_template_matching)
        self.assertEqual(benchmark.warmup_frames, 5)
    
    def test_performance_profiler(self):
        """Test performance profiling functionality"""
        profiler = PerformanceProfiler()
        
        profiler.start()
        time.sleep(0.1)  # Simulate work
        elapsed, cpu, memory = profiler.stop()
        
        self.assertGreater(elapsed, 0.05)  # Should be at least 50ms
        self.assertGreaterEqual(cpu, 0)
        # Memory delta can be positive or negative
    
    def test_benchmark_metrics_dataclass(self):
        """Test BenchmarkMetrics dataclass"""
        metrics = BenchmarkMetrics(
            method=DetectionMethod.YOLO,
            detection_time=0.05,
            fps=20.0,
            cpu_usage=15.5,
            memory_usage=128.0,
            gpu_usage=45.0,
            cards_detected=5,
            chips_detected=3,
            players_detected=2,
            markers_detected=1,
            total_objects_detected=11
        )
        
        self.assertEqual(metrics.method, DetectionMethod.YOLO)
        self.assertEqual(metrics.detection_time, 0.05)
        self.assertEqual(metrics.fps, 20.0)
        self.assertEqual(metrics.total_objects_detected, 11)
    
    def test_create_test_images(self):
        """Test synthetic test image creation"""
        benchmark = DetectionBenchmark(enable_yolo=False, enable_template_matching=False)
        
        images = benchmark.create_test_images(10)
        
        self.assertEqual(len(images), 10)
        for image in images:
            self.assertEqual(len(image.shape), 3)  # Height, width, channels
            self.assertGreater(image.shape[0], 0)  # Height > 0
            self.assertGreater(image.shape[1], 0)  # Width > 0
            self.assertEqual(image.shape[2], 3)     # 3 channels (BGR)
    
    @patch('DetectionBenchmark.YOLOPokerDetector')
    def test_speed_benchmark_mock(self, mock_yolo_class):
        """Test speed benchmark with mocked YOLO detector"""
        # Setup mock YOLO detector
        mock_detector = Mock()
        mock_detection_result = Mock()
        mock_detection_result.cards = []
        mock_detection_result.chips = []
        mock_detection_result.players = []
        mock_detection_result.markers = []
        mock_detection_result.detection_time = 0.02
        
        mock_detector.detect_poker_objects.return_value = mock_detection_result
        mock_yolo_class.return_value = mock_detector
        
        benchmark = DetectionBenchmark(enable_template_matching=False)
        
        result = benchmark.run_speed_benchmark(self.test_images)
        
        self.assertEqual(result.test_name, "Speed Benchmark")
        self.assertEqual(result.image_count, len(self.test_images))
        self.assertIn(DetectionMethod.YOLO, result.metrics_by_method)
    
    def test_detection_method_enum(self):
        """Test DetectionMethod enum"""
        self.assertEqual(DetectionMethod.YOLO.value, "yolo")
        self.assertEqual(DetectionMethod.TEMPLATE_MATCHING.value, "template_matching")
        self.assertEqual(DetectionMethod.CNN_BASED.value, "cnn_based")
    
    @patch('DetectionBenchmark.YOLOPokerDetector')
    def test_scalability_benchmark(self, mock_yolo_class):
        """Test scalability benchmark across different resolutions"""
        # Setup mock detector
        mock_detector = Mock()
        mock_detection_result = Mock()
        mock_detection_result.cards = []
        mock_detection_result.chips = []
        mock_detection_result.players = []
        mock_detection_result.markers = []
        mock_detection_result.detection_time = 0.03
        
        mock_detector.detect_poker_objects.return_value = mock_detection_result
        mock_yolo_class.return_value = mock_detector
        
        benchmark = DetectionBenchmark(enable_template_matching=False)
        
        # Test different resolutions
        resolutions = [(640, 480), (1280, 720), (1920, 1080)]
        result = benchmark.run_scalability_benchmark(resolutions)
        
        self.assertEqual(result.test_name, "Scalability Benchmark")
        self.assertEqual(result.image_count, len(resolutions))
    
    def test_save_benchmark_results(self):
        """Test saving benchmark results to file"""
        # Create a minimal benchmark with mock data
        benchmark = DetectionBenchmark(enable_yolo=False, enable_template_matching=False)
        
        # Add mock result
        from DetectionBenchmark import BenchmarkResult
        mock_result = BenchmarkResult(
            test_name="Test Benchmark",
            image_count=5,
            total_time=1.0,
            metrics_by_method={},
            summary_stats={"test_method": {"avg_detection_time": 0.2}}
        )
        benchmark.results.append(mock_result)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            benchmark.save_results(f.name)
            
            # Verify file was created and contains data
            with open(f.name, 'r') as read_file:
                data = json.load(read_file)
                self.assertEqual(len(data), 1)
                self.assertEqual(data[0]['test_name'], "Test Benchmark")
    
    def test_performance_report_generation(self):
        """Test performance report generation"""
        benchmark = DetectionBenchmark(enable_yolo=False, enable_template_matching=False)
        
        # Add mock result with summary stats
        from DetectionBenchmark import BenchmarkResult
        mock_result = BenchmarkResult(
            test_name="Performance Test",
            image_count=10,
            total_time=2.0,
            metrics_by_method={},
            summary_stats={
                "yolo": {
                    "avg_detection_time": 0.05,
                    "avg_fps": 20.0,
                    "avg_cpu_usage": 25.0,
                    "total_objects_detected": 50
                },
                "template_matching": {
                    "avg_detection_time": 0.1,
                    "avg_fps": 10.0,
                    "avg_cpu_usage": 15.0,
                    "total_objects_detected": 45
                }
            }
        )
        benchmark.results.append(mock_result)
        
        report = benchmark.generate_performance_report()
        
        self.assertIn("POKER DETECTION BENCHMARK REPORT", report)
        self.assertIn("Performance Test", report)
        self.assertIn("YOLO", report)
        self.assertIn("RECOMMENDATIONS", report)

class TestIntegrationBenchmark(unittest.TestCase):
    """Integration tests for YOLO + Benchmarking system"""
    
    def test_quick_benchmark_function(self):
        """Test the convenience quick benchmark function"""
        with patch('DetectionBenchmark.DetectionBenchmark') as mock_benchmark_class:
            # Setup mock benchmark
            mock_benchmark = Mock()
            mock_benchmark.create_test_images.return_value = [np.zeros((480, 640, 3), dtype=np.uint8)]
            mock_benchmark.run_speed_benchmark.return_value = Mock()
            mock_benchmark.generate_performance_report.return_value = "Mock Report"
            mock_benchmark.save_results.return_value = None
            
            mock_benchmark_class.return_value = mock_benchmark
            
            report = run_quick_benchmark(num_images=5)
            
            self.assertEqual(report, "Mock Report")
            mock_benchmark.create_test_images.assert_called_once_with(5)
            mock_benchmark.run_speed_benchmark.assert_called_once()
    
    @patch('YOLODetection.torch')
    def test_yolo_factory_function(self, mock_torch):
        """Test YOLO detector factory function"""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.hub.load.return_value = Mock()
        
        detector = create_poker_yolo_detector(use_gpu=False)
        
        self.assertIsInstance(detector, YOLOPokerDetector)
    
    def test_benchmark_with_mock_yolo(self):
        """Test full benchmark workflow with mocked components"""
        with patch('DetectionBenchmark.YOLOPokerDetector') as mock_yolo, \
             patch('DetectionBenchmark.ComputerVisionEngine') as mock_cv:
            
            # Setup YOLO mock
            mock_yolo_instance = Mock()
            mock_yolo_detection = Mock()
            mock_yolo_detection.cards = []
            mock_yolo_detection.chips = []
            mock_yolo_detection.players = []
            mock_yolo_detection.markers = []
            mock_yolo_detection.detection_time = 0.03
            mock_yolo_instance.detect_poker_objects.return_value = mock_yolo_detection
            mock_yolo.return_value = mock_yolo_instance
            
            # Setup CV mock  
            mock_cv_instance = Mock()
            mock_cv_detection = Mock()
            mock_cv_detection.player_cards = []
            mock_cv_detection.board_cards = []
            mock_cv_instance.detect_game_state.return_value = mock_cv_detection
            mock_cv.return_value = mock_cv_instance
            
            # Run benchmark
            benchmark = DetectionBenchmark()
            test_images = benchmark.create_test_images(3)
            result = benchmark.run_speed_benchmark(test_images)
            
            # Verify results
            self.assertIsNotNone(result)
            self.assertEqual(result.image_count, 3)
            self.assertIn(DetectionMethod.YOLO, result.metrics_by_method)
            self.assertIn(DetectionMethod.TEMPLATE_MATCHING, result.metrics_by_method)

class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def test_yolo_initialization_fallback(self):
        """Test YOLO initialization fallback to mock when models unavailable"""
        with patch('YOLODetection.torch') as mock_torch:
            # Simulate torch unavailable
            mock_torch.hub.load.side_effect = Exception("Model not found")
            
            detector = YOLOPokerDetector(use_gpu=False)
            
            # Should fall back to mock mode
            self.assertTrue(hasattr(detector, 'use_mock'))
    
    def test_benchmark_with_no_detectors(self):
        """Test benchmark behavior when no detectors are available"""
        benchmark = DetectionBenchmark(
            enable_yolo=False, 
            enable_template_matching=False, 
            enable_cnn=False
        )
        
        test_images = [np.zeros((480, 640, 3), dtype=np.uint8)]
        result = benchmark.run_speed_benchmark(test_images)
        
        # Should handle gracefully
        self.assertEqual(len(result.metrics_by_method), 0)
    
    def test_empty_detection_results(self):
        """Test handling of empty detection results"""
        detector = YOLOPokerDetector(use_gpu=False)
        detector.use_mock = True
        
        # Mock empty detection
        with patch.object(detector, '_mock_detection', return_value=[]):
            result = detector.detect_poker_objects(np.zeros((480, 640, 3), dtype=np.uint8))
            
            self.assertEqual(len(result.cards), 0)
            self.assertEqual(len(result.chips), 0)
            self.assertGreater(result.detection_time, 0)

if __name__ == '__main__':
    # Configure test logging
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise during tests
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestYOLODetection,
        TestDetectionBenchmark, 
        TestIntegrationBenchmark,
        TestErrorHandling
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"🧪 YOLO INTEGRATION TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors))/result.testsRun*100:.1f}%")
    
    if result.failures:
        print(f"\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip() if 'AssertionError:' in traceback else 'Unknown failure'}")
    
    if result.errors:
        print(f"\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip() if 'Exception:' in traceback else 'Unknown error'}")
    
    if not result.failures and not result.errors:
        print(f"\n✅ All tests passed! YOLO integration is ready for production.")
    else:
        print(f"\n⚠️  Some tests failed. Please review the issues before deployment.")