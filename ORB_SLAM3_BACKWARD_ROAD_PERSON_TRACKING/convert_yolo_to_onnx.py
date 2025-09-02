#!/usr/bin/env python3
"""
Convert YOLOv10 PyTorch model to ONNX format for C++ inference
"""

import torch
import torch.onnx
import os
import sys

def convert_yolo_to_onnx():
    """Convert YOLOv10 PyTorch model to ONNX format"""
    
    # Model paths - Use YOLOv10m for better accuracy (same as Python version)
    pt_model_path = r"D:\Learning\realtimesystem\yolov10m.pt"
    onnx_model_path = r"D:\Learning\ORB_SLAM3_macosx\yolov10m.onnx"
    
    print(f"Converting {pt_model_path} to {onnx_model_path}")
    
    # Check if PyTorch model exists
    if not os.path.exists(pt_model_path):
        print(f"ERROR: PyTorch model not found: {pt_model_path}")
        return False
    
    try:
        # Load the YOLOv10 model
        print("Loading PyTorch model...")
        
        # Try different loading methods
        model = None
        try:
            # Method 1: Try YOLOv10 specific loading first
            from ultralytics import YOLO
            yolo = YOLO(pt_model_path)
            model = yolo.model.float()  # Ensure float32
            print("SUCCESS: Loaded using ultralytics YOLO")
        except ImportError:
            print("WARNING: ultralytics not available, trying direct torch.load")
            try:
                # Method 2: Direct torch.load
                checkpoint = torch.load(pt_model_path, map_location='cpu', weights_only=False)
                if hasattr(checkpoint, 'model'):
                    model = checkpoint.model.float()
                elif isinstance(checkpoint, dict) and 'model' in checkpoint:
                    model = checkpoint['model'].float()
                else:
                    model = checkpoint.float()
                print("SUCCESS: Loaded using torch.load")
            except Exception as e:
                print(f"ERROR: All loading methods failed: {e}")
                return False
        except Exception as e:
            print(f"WARNING: ultralytics failed: {e}, trying torch.load")
            try:
                # Method 2: Direct torch.load
                checkpoint = torch.load(pt_model_path, map_location='cpu', weights_only=False)
                if hasattr(checkpoint, 'model'):
                    model = checkpoint.model.float()
                elif isinstance(checkpoint, dict) and 'model' in checkpoint:
                    model = checkpoint['model'].float()
                else:
                    model = checkpoint.float()
                print("SUCCESS: Loaded using torch.load")
            except Exception as e:
                print(f"ERROR: All loading methods failed: {e}")
                return False
        
        # Set model to evaluation mode
        model.eval()
        
        # Create dummy input tensor (batch_size=1, channels=3, height=640, width=640)
        dummy_input = torch.randn(1, 3, 640, 640, dtype=torch.float32)
        
        print("Converting to ONNX...")
        
        # Export to ONNX
        torch.onnx.export(
            model,                          # Model to export
            dummy_input,                    # Dummy input
            onnx_model_path,               # Output path
            export_params=True,            # Store trained parameters
            opset_version=11,              # ONNX opset version
            do_constant_folding=True,      # Constant folding optimization
            input_names=['images'],        # Input tensor names
            output_names=['output'],       # Output tensor names
            dynamic_axes={
                'images': {0: 'batch_size'},    # Variable batch size
                'output': {0: 'batch_size'}     # Variable batch size
            }
        )
        
        print(f"SUCCESS: Model converted successfully!")
        print(f"ONNX model saved to: {onnx_model_path}")
        
        # Verify the ONNX model
        try:
            import onnx
            onnx_model = onnx.load(onnx_model_path)
            onnx.checker.check_model(onnx_model)
            print("SUCCESS: ONNX model verification passed")
            
            # Print model info
            print(f"Model info:")
            print(f"   Inputs: {[input.name for input in onnx_model.graph.input]}")
            print(f"   Outputs: {[output.name for output in onnx_model.graph.output]}")
            
        except ImportError:
            print("WARNING: onnx package not available for verification")
        except Exception as e:
            print(f"WARNING: ONNX verification warning: {e}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Conversion failed: {e}")
        print(f"Try installing required packages:")
        print(f"   pip install torch torchvision ultralytics onnx")
        return False

if __name__ == "__main__":
    success = convert_yolo_to_onnx()
    sys.exit(0 if success else 1)