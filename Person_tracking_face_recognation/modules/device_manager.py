"""
Device and Camera Management Module
Handles GPU detection, YOLO model initialization, and camera setup
"""

import cv2
import numpy as np
import torch

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class DeviceManager:
    """Manages GPU/CPU devices, YOLO models, and camera initialization"""
    
    def __init__(self, config):
        self.config = config
        self.device = 'cpu'
        self.model = None
        self.camera_index = config.get('camera_index', 0)
        
        # HOG fallback detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Initialize YOLO with fallbacks
        self.init_yolo_with_fallback()
    
    def init_yolo_with_fallback(self):
        """Initialize YOLO with comprehensive error handling and fallbacks"""
        if not YOLO_AVAILABLE:
            print("⚠️ YOLO unavailable, using OpenCV HOG detector")
            return
        
        try:
            print("🚀 Initializing YOLO with fallbacks...")
            
            # Smart device detection
            self.device = self.detect_best_device()
            
            # Load model with error handling
            model_path = self.config.get('model', 'yolov8n.pt')
            self.model = YOLO(model_path)
            
            # Optimized warmup
            self.warmup_model()
            
            print(f"✅ YOLO ready on {self.device.upper()}")
            
        except Exception as e:
            print(f"❌ YOLO initialization failed: {e}")
            print("🔄 Falling back to CPU mode...")
            self.fallback_to_cpu()
    
    def detect_best_device(self):
        """Detect and configure the best available device"""
        if torch.cuda.is_available():
            try:
                # Test GPU with small operation
                test_tensor = torch.zeros(1, 3, 640, 640).cuda()
                del test_tensor
                torch.cuda.empty_cache()
                
                device = 'cuda'
                print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
                print(f"📊 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
                
                # GPU optimizations
                torch.backends.cudnn.benchmark = True
                torch.cuda.empty_cache()
                return device
            except Exception as e:
                print(f"⚠️ GPU test failed: {e}")
        
        print("🖥️ Using CPU")
        return 'cpu'
    
    def warmup_model(self):
        """Warm up model with error handling"""
        try:
            print("🔥 Warming up model...")
            dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
            
            for i in range(3):
                try:
                    self.model.predict(dummy_img, device=self.device, verbose=False)
                except Exception as e:
                    if i == 0:  # First attempt failed, try CPU
                        print(f"⚠️ GPU warmup failed, trying CPU: {e}")
                        self.device = 'cpu'
                    else:
                        print(f"⚠️ Warmup attempt {i+1} failed: {e}")
        except Exception as e:
            print(f"⚠️ Model warmup failed: {e}")
    
    def fallback_to_cpu(self):
        """Fallback to CPU mode with simpler model"""
        try:
            self.device = 'cpu'
            self.model = YOLO('yolov8n.pt')  # Use lighter model on CPU
            print("✅ Fallback to CPU successful")
        except Exception as e:
            print(f"❌ CPU fallback failed: {e}")
            self.model = None
    
    def initialize_camera_safe(self):
        """Safe camera initialization with multiple fallbacks"""
        cameras_to_try = [self.camera_index, 0, 1, 2]  # Try configured, then common indices
        
        for cam_idx in cameras_to_try:
            try:
                print(f"🔍 Trying camera index {cam_idx}...")
                # Use DirectShow backend on Windows for better compatibility
                cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
                
                if not cap.isOpened():
                    cap.release()
                    continue
                
                # Test camera with a frame read
                ret, test_frame = cap.read()
                if not ret or test_frame is None:
                    print(f"⚠️ Camera {cam_idx} opened but no frames")
                    cap.release()
                    continue
                
                # Configure camera settings
                try:
                    width = self.config.get('resolution', {}).get('width', 1280)
                    height = self.config.get('resolution', {}).get('height', 720)
                    
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    
                    # Verify settings
                    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
                    
                    print(f"✅ Camera {cam_idx} initialized successfully")
                    print(f"📺 Resolution: {actual_width}x{actual_height} @ {actual_fps}FPS")
                    
                    # Update camera index if different from config
                    if cam_idx != self.camera_index:
                        print(f"🔧 Using camera {cam_idx} instead of {self.camera_index}")
                        self.camera_index = cam_idx
                    
                    return cap
                    
                except Exception as setting_error:
                    print(f"⚠️ Camera {cam_idx} settings failed: {setting_error}")
                    # Continue with default settings
                    return cap
                    
            except Exception as e:
                print(f"⚠️ Camera {cam_idx} failed: {e}")
                if cap is not None:
                    cap.release()
                continue
        
        print("❌ No working cameras found")
        return None
    
    def detect_persons_optimized(self, frame):
        """Enhanced person detection with comprehensive error handling"""
        try:
            if self.model is None:
                return self.detect_persons_hog(frame)
            
            # Validate frame
            if frame is None or frame.size == 0:
                return [], []
            
            # GPU memory management
            if self.device == 'cuda' and torch.cuda.is_available():
                try:
                    memory_available = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
                    if memory_available < 100 * 1024 * 1024:  # Less than 100MB
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            
            # Safe prediction with fallbacks
            try:
                results = self.model.predict(
                    frame, 
                    device=self.device, 
                    verbose=False,
                    half=True if self.device == 'cuda' else False,
                    imgsz=640,
                    max_det=self.config.get('max_detections', 10),
                    conf=self.config.get('confidence_threshold', 0.4)
                )
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print("⚠️ GPU memory full, using CPU...")
                    torch.cuda.empty_cache()
                    results = self.model.predict(
                        frame, device='cpu', verbose=False, 
                        conf=self.config.get('confidence_threshold', 0.4)
                    )
                    self.device = 'cpu'  # Switch to CPU for future predictions
                else:
                    raise e
            
            # Safe result processing
            detections = []
            confidences = []
            
            try:
                for r in results:
                    if r.boxes is not None:
                        for box in r.boxes:
                            try:
                                class_id = int(box.cls[0])
                                confidence = float(box.conf[0])
                                
                                if class_id == 0 and confidence >= self.config.get('confidence_threshold', 0.4):
                                    bbox = box.xyxy[0].tolist()
                                    # Validate bbox values
                                    if (len(bbox) == 4 and 
                                        all(isinstance(x, (int, float)) and not np.isnan(x) for x in bbox)):
                                        detections.append(bbox)
                                        confidences.append(confidence)
                            except (IndexError, ValueError, TypeError):
                                continue  # Skip invalid detections
            except Exception as e:
                print(f"⚠️ Result processing error: {e}")
            
            return detections, confidences
            
        except Exception as e:
            print(f"🔄 Detection failed, using HOG fallback: {e}")
            # Handle different error types
            if any(keyword in str(e).lower() for keyword in ['cuda', 'gpu', 'device', 'tensor']):
                print("🔧 GPU issue detected, switching to CPU...")
                self.device = 'cpu'
                if self.model is not None:
                    try:
                        return self.detect_persons_optimized(frame)  # Retry on CPU
                    except:
                        pass  # Fall through to HOG
            
            self.model = None  # Disable YOLO to prevent repeated errors
            return self.detect_persons_hog(frame)
    
    def detect_persons_hog(self, frame):
        """Fallback HOG detection with error handling"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            boxes, weights = self.hog.detectMultiScale(
                gray, winStride=(8, 8), padding=(32, 32), scale=1.05
            )
            
            detections = []
            for (x, y, w, h) in boxes:
                detections.append([x, y, x + w, y + h])
            
            return detections, [0.8] * len(detections)
            
        except Exception as e:
            print(f"HOG detection error: {e}")
            return [], []
    
    def get_device_info(self):
        """Get current device information"""
        return {
            'device': self.device,
            'model_available': self.model is not None,
            'yolo_available': YOLO_AVAILABLE,
            'gpu_available': torch.cuda.is_available() if YOLO_AVAILABLE else False
        }