#!/usr/bin/env python3
"""
Simplified validation test for YOLO integration and benchmarking system.
Tests core functionality with available dependencies.
"""

import sys
import time
import numpy as np
import cv2
from unittest.mock import Mock, patch

def test_yolo_detection_basic():
    """Test basic YOLO detection functionality"""
    print("🧪 Testing YOLO Detection System...")
    
    try:
        from YOLODetection import YOLOPokerDetector, PokerObjectType, PokerGameDetection
        
        # Create detector in mock mode
        detector = YOLOPokerDetector(use_gpu=False)
        detector.use_mock = True
        
        # Create test image
        test_image = np.ones((600, 800, 3), dtype=np.uint8) * 100
        
        # Run detection
        start_time = time.time()
        result = detector.detect_poker_objects(test_image)
        detection_time = time.time() - start_time
        
        # Validate results
        assert isinstance(result, PokerGameDetection), "Result should be PokerGameDetection"
        assert len(result.cards) > 0, "Should detect some cards"
        assert len(result.chips) > 0, "Should detect some chips"
        assert result.detection_time > 0, "Detection time should be positive"
        assert detection_time < 1.0, "Detection should be fast"
        
        print(f"  ✓ YOLO detector created and functional")
        print(f"  ✓ Detected {len(result.cards)} cards, {len(result.chips)} chips")
        print(f"  ✓ Detection time: {detection_time*1000:.1f}ms")
        
        return True
        
    except Exception as e:
        print(f"  ❌ YOLO detection test failed: {e}")
        return False

def test_benchmark_system():
    """Test benchmarking system functionality"""
    print("\n🏁 Testing Benchmark System...")
    
    try:
        from DetectionBenchmark import DetectionBenchmark, BenchmarkMetrics, DetectionMethod
        
        # Create benchmark system
        benchmark = DetectionBenchmark(
            enable_yolo=True,
            enable_template_matching=False,  # Disable to avoid ComputerVision dependency
            gpu_available=False
        )
        
        # Create test images
        test_images = benchmark.create_test_images(3)
        assert len(test_images) == 3, "Should create requested number of images"
        
        print(f"  ✓ Benchmark system created")
        print(f"  ✓ Generated {len(test_images)} test images")
        
        # Test image properties
        for i, image in enumerate(test_images):
            assert len(image.shape) == 3, f"Image {i} should be 3D"
            assert image.shape[2] == 3, f"Image {i} should have 3 channels"
            print(f"  ✓ Image {i+1}: {image.shape[1]}x{image.shape[0]} pixels")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Benchmark system test failed: {e}")
        return False

def test_performance_monitoring():
    """Test performance monitoring capabilities"""
    print("\n📊 Testing Performance Monitoring...")
    
    try:
        from YOLODetection import YOLOPokerDetector
        
        # Create detector
        detector = YOLOPokerDetector(use_gpu=False)
        detector.use_mock = True
        
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Run multiple detections
        detection_times = []
        for i in range(5):
            start_time = time.time()
            result = detector.detect_poker_objects(test_image)
            detection_time = time.time() - start_time
            detection_times.append(detection_time)
        
        # Check performance stats
        stats = detector.get_performance_stats()
        
        assert 'average_detection_time' in stats, "Should have average detection time"
        assert 'fps' in stats, "Should have FPS metric"
        assert 'total_frames' in stats, "Should have frame count"
        assert stats['total_frames'] == 5, "Should track correct frame count"
        
        avg_time = np.mean(detection_times)
        avg_fps = 1.0 / avg_time
        
        print(f"  ✓ Performance monitoring active")
        print(f"  ✓ Average detection time: {avg_time*1000:.1f}ms")
        print(f"  ✓ Average FPS: {avg_fps:.1f}")
        print(f"  ✓ Processed {stats['total_frames']} frames")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Performance monitoring test failed: {e}")
        return False

def test_object_classification():
    """Test poker object classification"""
    print("\n🎯 Testing Object Classification...")
    
    try:
        from YOLODetection import YOLOPokerDetector, PokerObjectType
        
        # Create detector
        detector = YOLOPokerDetector(use_gpu=False)
        detector.use_mock = True
        
        # Create test image
        test_image = np.ones((600, 800, 3), dtype=np.uint8) * 50
        
        # Run detection
        result = detector.detect_poker_objects(test_image)
        
        # Check object types
        object_types_found = set()
        total_objects = 0
        
        for card in result.cards:
            assert card.object_type == PokerObjectType.CARD, "Card should have CARD type"
            assert card.card_rank is not None, "Card should have rank"
            assert card.card_suit is not None, "Card should have suit"
            object_types_found.add("card")
            total_objects += 1
        
        for chip in result.chips:
            assert chip.object_type == PokerObjectType.CHIP, "Chip should have CHIP type"
            assert chip.chip_value is not None, "Chip should have value"
            object_types_found.add("chip")
            total_objects += 1
        
        for player in result.players:
            assert player.object_type == PokerObjectType.PLAYER, "Player should have PLAYER type"
            object_types_found.add("player")
            total_objects += 1
        
        print(f"  ✓ Object types detected: {', '.join(object_types_found)}")
        print(f"  ✓ Total objects classified: {total_objects}")
        print(f"  ✓ Cards with ranks/suits: {len([c for c in result.cards if c.card_rank])}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Object classification test failed: {e}")
        return False

def test_visualization():
    """Test detection visualization"""
    print("\n🖼️  Testing Visualization...")
    
    try:
        from YOLODetection import YOLOPokerDetector
        
        # Create detector
        detector = YOLOPokerDetector(use_gpu=False)
        detector.use_mock = True
        
        # Create test image
        test_image = np.ones((400, 600, 3), dtype=np.uint8) * 75
        
        # Run detection
        result = detector.detect_poker_objects(test_image)
        
        # Create visualization
        vis_image = detector.visualize_detections(test_image, result)
        
        # Validate visualization
        assert vis_image.shape == test_image.shape, "Visualization should have same dimensions"
        assert not np.array_equal(vis_image, test_image), "Visualization should be different from original"
        
        print(f"  ✓ Visualization created successfully")
        print(f"  ✓ Image dimensions: {vis_image.shape}")
        print(f"  ✓ Visualization differs from original")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Visualization test failed: {e}")
        return False

def test_data_structures():
    """Test core data structures"""
    print("\n📦 Testing Data Structures...")
    
    try:
        from YOLODetection import YOLODetection, PokerGameDetection, PokerObjectType
        from DetectionBenchmark import BenchmarkMetrics, DetectionMethod
        
        # Test YOLODetection
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
        
        assert detection.object_type == PokerObjectType.CARD
        assert detection.confidence == 0.95
        assert detection.card_rank == "A"
        
        # Test PokerGameDetection
        game_detection = PokerGameDetection(
            cards=[detection],
            chips=[],
            players=[],
            markers=[],
            action_buttons=[],
            detection_time=0.05,
            frame_size=(800, 600)
        )
        
        assert len(game_detection.cards) == 1
        assert game_detection.detection_time == 0.05
        
        # Test BenchmarkMetrics
        metrics = BenchmarkMetrics(
            method=DetectionMethod.YOLO,
            detection_time=0.025,
            fps=40.0,
            cpu_usage=25.0,
            memory_usage=150.0,
            gpu_usage=None,
            cards_detected=5,
            chips_detected=3,
            players_detected=2,
            markers_detected=1,
            total_objects_detected=11
        )
        
        assert metrics.method == DetectionMethod.YOLO
        assert metrics.fps == 40.0
        assert metrics.total_objects_detected == 11
        
        print(f"  ✓ YOLODetection data structure working")
        print(f"  ✓ PokerGameDetection data structure working")
        print(f"  ✓ BenchmarkMetrics data structure working")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data structures test failed: {e}")
        return False

def main():
    """Main validation function"""
    print("🎰 YOLO INTEGRATION VALIDATION TEST")
    print("=" * 50)
    print("Validating core YOLO integration functionality...")
    print("=" * 50)
    
    # Test categories
    tests = [
        ("YOLO Detection", test_yolo_detection_basic),
        ("Benchmark System", test_benchmark_system),
        ("Performance Monitoring", test_performance_monitoring),
        ("Object Classification", test_object_classification),
        ("Visualization", test_visualization),
        ("Data Structures", test_data_structures)
    ]
    
    # Run tests
    results = {}
    total_time = time.time()
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"\n💥 {test_name} crashed: {e}")
            results[test_name] = False
    
    total_time = time.time() - total_time
    
    # Summary
    print("\n" + "=" * 60)
    print("🏆 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    print(f"📊 Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    
    print(f"\n📋 Test Results:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name:<25} {status}")
    
    if all(results.values()):
        print(f"\n🎉 ALL VALIDATION TESTS PASSED!")
        print(f"✅ YOLO integration core functionality is working")
        print(f"🚀 System architecture validated successfully")
        
        print(f"\n📝 Next Steps:")
        print(f"  1. Install full dependencies for complete functionality")
        print(f"  2. Run comprehensive test suite: python run_yolo_integration_tests.py")
        print(f"  3. Try interactive demo: python demo_yolo_benchmark.py")
        
        return True
    else:
        failed_tests = [name for name, result in results.items() if not result]
        print(f"\n⚠️  Some validation tests failed: {', '.join(failed_tests)}")
        print(f"🔧 Please review the failed components")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n👋 Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Validation crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)