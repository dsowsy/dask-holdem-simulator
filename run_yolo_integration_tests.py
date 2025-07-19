#!/usr/bin/env python3
"""
Comprehensive test runner for YOLO integration and benchmarking system.
Runs all tests and demonstrates the integrated poker detection capabilities.
"""

import sys
import os
import time
import subprocess
import traceback
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    missing_deps = []
    required_packages = [
        'numpy', 'cv2', 'torch', 'matplotlib', 'psutil'
    ]
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'torch':
                import torch
            elif package == 'numpy':
                import numpy
            elif package == 'matplotlib':
                import matplotlib
            elif package == 'psutil':
                import psutil
            print(f"  ✓ {package}")
        except ImportError:
            missing_deps.append(package)
            print(f"  ❌ {package} - MISSING")
    
    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("Installing missing dependencies...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_deps)
            print("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies. Tests will run in mock mode.")
            return False
    else:
        print("✅ All dependencies available")
    
    return True

def run_unit_tests():
    """Run unit tests for YOLO integration"""
    print("\n🧪 Running YOLO Integration Unit Tests")
    print("=" * 50)
    
    try:
        # Import and run the test suite
        from test_yolo_integration import (
            TestYOLODetection, TestDetectionBenchmark, 
            TestIntegrationBenchmark, TestErrorHandling
        )
        import unittest
        
        # Create test suite
        test_suite = unittest.TestSuite()
        
        test_classes = [
            TestYOLODetection,
            TestDetectionBenchmark,
            TestIntegrationBenchmark, 
            TestErrorHandling
        ]
        
        for test_class in test_classes:
            tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
            test_suite.addTests(tests)
        
        # Run tests with detailed output
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        result = runner.run(test_suite)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"🧪 UNIT TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        
        if result.failures:
            print(f"\n❌ FAILURES ({len(result.failures)}):")
            for test, traceback in result.failures:
                print(f"  - {test}")
        
        if result.errors:
            print(f"\n💥 ERRORS ({len(result.errors)}):")
            for test, traceback in result.errors:
                print(f"  - {test}")
        
        success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
        print(f"\n📊 Success Rate: {success_rate:.1f}%")
        
        return len(result.failures) == 0 and len(result.errors) == 0
        
    except Exception as e:
        print(f"❌ Failed to run unit tests: {e}")
        traceback.print_exc()
        return False

def run_integration_demo():
    """Run integration demonstration"""
    print("\n🎮 Running Integration Demonstration")
    print("=" * 50)
    
    try:
        # Import demo modules
        from YOLODetection import create_poker_yolo_detector
        from DetectionBenchmark import DetectionBenchmark
        from ComputerVision import ComputerVisionEngine
        
        print("  🔧 Initializing systems...")
        
        # Initialize YOLO detector
        yolo_detector = create_poker_yolo_detector(use_gpu=False)
        print("    ✓ YOLO detector initialized")
        
        # Initialize benchmark system
        benchmark = DetectionBenchmark(
            enable_yolo=True,
            enable_template_matching=True,
            gpu_available=False
        )
        print("    ✓ Benchmark system initialized")
        
        # Create test images
        print("  🎨 Creating test scenarios...")
        test_images = benchmark.create_test_images(5)
        print(f"    ✓ Created {len(test_images)} test images")
        
        # Test YOLO detection
        print("  🎯 Testing YOLO detection...")
        total_detection_time = 0
        total_objects = 0
        
        for i, image in enumerate(test_images):
            start_time = time.time()
            result = yolo_detector.detect_poker_objects(image)
            detection_time = time.time() - start_time
            
            total_detection_time += detection_time
            objects_found = len(result.cards) + len(result.chips) + len(result.players) + len(result.markers)
            total_objects += objects_found
            
            print(f"    Image {i+1}: {detection_time*1000:.1f}ms, {objects_found} objects")
        
        avg_detection_time = total_detection_time / len(test_images)
        avg_fps = 1.0 / avg_detection_time
        
        print(f"  📊 YOLO Performance Summary:")
        print(f"    Average detection time: {avg_detection_time*1000:.2f}ms")
        print(f"    Average FPS: {avg_fps:.1f}")
        print(f"    Total objects detected: {total_objects}")
        
        # Run quick benchmark
        print("  🏁 Running performance benchmark...")
        benchmark_result = benchmark.run_speed_benchmark(test_images[:3])  # Use fewer images for speed
        
        if benchmark_result.summary_stats:
            print("  📈 Benchmark Results:")
            for method, stats in benchmark_result.summary_stats.items():
                print(f"    {method}: {stats['avg_detection_time']*1000:.2f}ms avg, {stats['avg_fps']:.1f} FPS")
        
        print("✅ Integration demonstration completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Integration demo failed: {e}")
        traceback.print_exc()
        return False

def run_performance_benchmark():
    """Run comprehensive performance benchmark"""
    print("\n🏁 Running Performance Benchmark")
    print("=" * 50)
    
    try:
        from DetectionBenchmark import run_quick_benchmark
        
        print("  🚀 Starting benchmark suite...")
        print("  (This may take a few minutes)")
        
        start_time = time.time()
        report = run_quick_benchmark(num_images=10)
        benchmark_time = time.time() - start_time
        
        print(f"\n📊 Benchmark completed in {benchmark_time:.1f} seconds")
        print("\n" + "="*80)
        print("📈 PERFORMANCE BENCHMARK RESULTS")
        print("="*80)
        print(report)
        
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        traceback.print_exc()
        return False

def run_visual_demo():
    """Run visual demonstration (if display available)"""
    print("\n🖼️  Visual Demo")
    print("=" * 50)
    
    try:
        import cv2
        
        # Check if display is available
        try:
            cv2.namedWindow('test', cv2.WINDOW_NORMAL)
            cv2.destroyWindow('test')
            display_available = True
        except:
            display_available = False
        
        if not display_available:
            print("  ⚠️  No display available, skipping visual demo")
            return True
        
        from demo_yolo_benchmark import YOLOPokerDemo
        
        print("  🎮 Starting interactive demo...")
        print("  (Running in automated mode for 10 seconds)")
        
        demo = YOLOPokerDemo()
        
        # Run demo for a few seconds automatically
        for i in range(4):  # Cycle through 4 scenes
            scene_name, scene_image = demo.demo_scenes[i]
            
            print(f"    Testing scene: {scene_name}")
            
            # Run detection
            result = demo.yolo_detector.detect_poker_objects(scene_image)
            
            # Create visualization
            vis_image = demo.yolo_detector.visualize_detections(scene_image, result)
            
            # Display briefly
            cv2.imshow('YOLO Demo', vis_image)
            cv2.waitKey(1000)  # Show for 1 second
            
            objects_found = len(result.cards) + len(result.chips) + len(result.players) + len(result.markers)
            print(f"      Objects detected: {objects_found}")
        
        cv2.destroyAllWindows()
        print("✅ Visual demo completed")
        return True
        
    except Exception as e:
        print(f"❌ Visual demo failed: {e}")
        traceback.print_exc()
        return False

def generate_summary_report(results):
    """Generate final summary report"""
    print("\n" + "="*80)
    print("🏆 YOLO INTEGRATION TEST SUMMARY REPORT")
    print("="*80)
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    
    print(f"📊 Overall Results:")
    print(f"  Total test categories: {total_tests}")
    print(f"  Passed: {passed_tests}")
    print(f"  Failed: {total_tests - passed_tests}")
    print(f"  Success rate: {passed_tests/total_tests*100:.1f}%")
    
    print(f"\n📋 Test Category Results:")
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:<25} {status}")
    
    if all(results.values()):
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ YOLO integration is working correctly")
        print(f"🚀 System is ready for production use")
    else:
        failed_tests = [name for name, passed in results.items() if not passed]
        print(f"\n⚠️  Some tests failed: {', '.join(failed_tests)}")
        print(f"🔧 Please review the failed components before deployment")
    
    print(f"\n📁 Generated Files:")
    generated_files = [
        "quick_benchmark_results.json",
        "demo_benchmark_results.json",
        "demo_plots/",
        "detection_results_*.json"
    ]
    
    for filename in generated_files:
        if '*' in filename or Path(filename).exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  - {filename} (not generated)")

def main():
    """Main test runner function"""
    print("🎰 YOLO POKER DETECTION INTEGRATION TEST SUITE")
    print("=" * 60)
    print("This comprehensive test suite validates the YOLO integration")
    print("and benchmarking system for poker object detection.")
    print("=" * 60)
    
    # Track results
    test_results = {}
    
    # Check dependencies first
    deps_ok = check_dependencies()
    if not deps_ok:
        print("⚠️  Running tests in mock mode due to missing dependencies")
    
    # Run test categories
    test_categories = [
        ("Unit Tests", run_unit_tests),
        ("Integration Demo", run_integration_demo),
        ("Performance Benchmark", run_performance_benchmark),
        ("Visual Demo", run_visual_demo)
    ]
    
    for test_name, test_func in test_categories:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            test_results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"\n{test_name}: {status}")
        except KeyboardInterrupt:
            print(f"\n👋 Test suite interrupted by user")
            test_results[test_name] = False
            break
        except Exception as e:
            print(f"\n💥 {test_name} crashed: {e}")
            test_results[test_name] = False
    
    # Generate final report
    generate_summary_report(test_results)
    
    # Exit with appropriate code
    if all(test_results.values()):
        print(f"\n🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        print(f"\n⚠️  Some tests failed. Please review the output above.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n👋 Test suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        traceback.print_exc()
        sys.exit(1)