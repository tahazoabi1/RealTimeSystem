import face_recognition
import cv2
import numpy as np
import pickle
import os
from datetime import datetime
import json


class FaceRecognizer:
    def __init__(self, face_db_path='face_database.pkl', min_face_size=60):
        self.face_db_path = face_db_path
        self.min_face_size = min_face_size
        self.known_face_encodings = []
        self.known_face_ids = []
        self.face_id_counter = 1
        self.face_locations_cache = {}
        self.face_names = {}  # Dictionary mapping face_id to name
        
        self.load_face_database()
        
    def load_face_database(self):
        if os.path.exists(self.face_db_path):
            try:
                with open(self.face_db_path, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data.get('encodings', [])
                    self.known_face_ids = data.get('ids', [])
                    self.face_id_counter = data.get('counter', 1)
                    self.face_names = data.get('names', {})
                print(f"Loaded {len(self.known_face_encodings)} known faces from database")
                if self.face_names:
                    print(f"Named faces: {self.face_names}")
            except Exception as e:
                print(f"Error loading face database: {e}")
                self.known_face_encodings = []
                self.known_face_ids = []
                self.face_id_counter = 1
                self.face_names = {}
        else:
            print("No existing face database found, starting fresh")
    
    def save_face_database(self):
        try:
            data = {
                'encodings': self.known_face_encodings,
                'ids': self.known_face_ids,
                'counter': self.face_id_counter,
                'names': self.face_names,
                'saved_at': datetime.now().isoformat()
            }
            with open(self.face_db_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"Face database saved with {len(self.known_face_encodings)} faces")
        except Exception as e:
            print(f"Error saving face database: {e}")
    
    def extract_face_from_bbox(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        
        # Add some padding around the face
        height, width = frame.shape[:2]
        padding = 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)
        
        face_img = frame[y1:y2, x1:x2]
        
        # Check minimum face size
        if face_img.shape[0] < self.min_face_size or face_img.shape[1] < self.min_face_size:
            return None, None
        
        return face_img, (x1, y1, x2, y2)
    
    def get_face_encoding(self, face_img):
        try:
            # Convert BGR to RGB
            rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Find face locations in the cropped image
            face_locations = face_recognition.face_locations(rgb_face, model="hog")
            
            if len(face_locations) > 0:
                # Get face encoding for the first (and hopefully only) face
                face_encodings = face_recognition.face_encodings(rgb_face, face_locations)
                if len(face_encodings) > 0:
                    return face_encodings[0]
            
            return None
        except Exception as e:
            print(f"Error getting face encoding: {e}")
            return None
    
    def recognize_face(self, face_encoding, tolerance=0.5):
        if face_encoding is None or len(self.known_face_encodings) == 0:
            return None
        
        try:
            # Compare with known faces
            distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            
            if len(distances) > 0:
                best_match_idx = np.argmin(distances)
                if distances[best_match_idx] <= tolerance:
                    return self.known_face_ids[best_match_idx]
            
            return None
        except Exception as e:
            print(f"Error recognizing face: {e}")
            return None
    
    def add_new_face(self, face_encoding):
        if face_encoding is not None:
            self.known_face_encodings.append(face_encoding)
            new_id = self.face_id_counter
            self.known_face_ids.append(new_id)
            self.face_id_counter += 1
            
            print(f"Added new face with ID: {new_id}")
            
            # Save database periodically
            if len(self.known_face_encodings) % 5 == 0:
                self.save_face_database()
            
            return new_id
        return None
    
    def process_detection(self, frame, bbox, track_id=None):
        try:
            # Extract face from bounding box
            face_img, adjusted_bbox = self.extract_face_from_bbox(frame, bbox)
            
            if face_img is None:
                return track_id  # Return original track_id if no face detected
            
            # Get face encoding
            face_encoding = self.get_face_encoding(face_img)
            
            if face_encoding is None:
                return track_id  # Return original track_id if encoding failed
            
            # Try to recognize the face
            recognized_id = self.recognize_face(face_encoding)
            
            if recognized_id is not None:
                return recognized_id
            else:
                # Add as new face
                new_id = self.add_new_face(face_encoding)
                return new_id if new_id is not None else track_id
                
        except Exception as e:
            print(f"Error processing face detection: {e}")
            return track_id
    
    def get_face_info(self):
        return {
            'total_known_faces': len(self.known_face_encodings),
            'next_id': self.face_id_counter
        }
    
    def reset_database(self):
        self.known_face_encodings = []
        self.known_face_ids = []
        self.face_id_counter = 1
        self.face_names = {}
        
        if os.path.exists(self.face_db_path):
            os.remove(self.face_db_path)
        
        print("Face database reset")
    
    def set_face_name(self, face_id, name):
        """Set a name for a specific face ID"""
        self.face_names[face_id] = name
        print(f"Face ID {face_id} named as '{name}'")
        self.save_face_database()
    
    def get_face_name(self, face_id):
        """Get the name for a face ID, returns ID if no name set"""
        return self.face_names.get(face_id, f"ID {face_id}")
    
    def list_faces_with_names(self):
        """List all faces with their names"""
        face_list = []
        for i, face_id in enumerate(self.known_face_ids):
            name = self.face_names.get(face_id, f"ID {face_id}")
            face_list.append({'id': face_id, 'name': name})
        return face_list
    
    def export_face_info(self, filename=None):
        if filename is None:
            filename = f'face_database_info_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        info = {
            'total_faces': len(self.known_face_encodings),
            'face_ids': self.known_face_ids,
            'exported_at': datetime.now().isoformat(),
            'database_file': self.face_db_path
        }
        
        with open(filename, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"Face database info exported to {filename}")
        return filename