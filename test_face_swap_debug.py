#!/usr/bin/env python3
"""
Debug script to test face swapping functionality and identify crash causes
"""

import os
import sys
import traceback

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    try:
        import roop.globals
        print("✓ roop.globals imported")
        
        import roop.core
        print("✓ roop.core imported")
        
        import roop.ProcessMgr
        print("✓ roop.ProcessMgr imported")
        
        import roop.processors.FaceSwapInsightFace
        print("✓ FaceSwapInsightFace imported")
        
        import cv2
        print("✓ cv2 imported")
        
        import numpy as np
        print("✓ numpy imported")
        
        import onnxruntime
        print("✓ onnxruntime imported")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_device_detection():
    """Test device detection and execution providers"""
    print("\nTesting device detection...")
    try:
        import roop.utilities as util
        device = util.get_device()
        print(f"✓ Detected device: {device}")
        
        import onnxruntime
        providers = onnxruntime.get_available_providers()
        print(f"✓ Available ONNX providers: {providers}")
        
        return True
    except Exception as e:
        print(f"✗ Device detection failed: {e}")
        traceback.print_exc()
        return False

def test_model_loading():
    """Test if face swap model can be loaded"""
    print("\nTesting model loading...")
    try:
        from roop.processors.FaceSwapInsightFace import FaceSwapInsightFace
        from roop.utilities import resolve_relative_path
        
        # Check if model file exists
        model_path = resolve_relative_path('../models/inswapper_128.onnx')
        if not os.path.exists(model_path):
            print(f"✗ Model file not found: {model_path}")
            return False
        
        print(f"✓ Model file found: {model_path}")
        
        # Try to initialize processor
        processor = FaceSwapInsightFace()
        options = {
            "devicename": "cpu",  # Use CPU for testing
            "modelname": "inswapper_128.onnx"
        }
        
        processor.Initialize(options)
        print("✓ FaceSwap processor initialized successfully")
        
        processor.Release()
        print("✓ FaceSwap processor released successfully")
        
        return True
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        traceback.print_exc()
        return False

def test_face_detection():
    """Test face detection functionality"""
    print("\nTesting face detection...")
    try:
        from roop.face_util import get_first_face
        import cv2
        import numpy as np
        
        # Create a simple test image
        test_image = np.zeros((256, 256, 3), dtype=np.uint8)
        test_image.fill(128)  # Gray image
        
        # Try face detection (should return None for gray image)
        face = get_first_face(test_image)
        print(f"✓ Face detection completed (result: {face})")
        
        return True
    except Exception as e:
        print(f"✗ Face detection failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=== ROOP Face Swap Debug Test ===\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Device Detection", test_device_detection),
        ("Model Loading", test_model_loading),
        ("Face Detection", test_face_detection)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    print("\n=== Test Summary ===")
    all_passed = True
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n✓ All tests passed! The issue might be in the UI or processing pipeline.")
    else:
        print("\n✗ Some tests failed. Check the errors above for clues about the crash.")
    
    print("\nTo run this test: python test_face_swap_debug.py")

if __name__ == "__main__":
    main()