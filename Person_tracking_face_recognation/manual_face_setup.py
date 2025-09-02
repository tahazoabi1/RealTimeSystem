#!/usr/bin/env python3
"""
Manual Face Setup Script
Allows you to manually add faces with specific IDs and names to the database
"""

import cv2
import pickle
import numpy as np
import face_recognition
from datetime import datetime
import os

def reset_database():
    """Reset the face database"""
    data = {
        'encodings': [],
        'ids': [],
        'counter': 1,
        'names': {},
        'saved_at': datetime.now().isoformat()
    }
    
    with open('face_database.pkl', 'wb') as f:
        pickle.dump(data, f)
    print("✅ Face database reset successfully")

def add_face_with_id_and_name(image_path, face_id, name):
    """Add a face with specific ID and name"""
    try:
        # Load and process image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return False
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Find faces in the image
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        if len(face_encodings) == 0:
            print(f"❌ No faces found in image: {image_path}")
            return False
        
        if len(face_encodings) > 1:
            print(f"⚠️ Multiple faces found, using the first one")
        
        # Use the first face encoding
        face_encoding = face_encodings[0]
        
        # Load existing database
        if os.path.exists('face_database.pkl'):
            with open('face_database.pkl', 'rb') as f:
                data = pickle.load(f)
        else:
            data = {
                'encodings': [],
                'ids': [],
                'counter': 1,
                'names': {},
                'saved_at': None
            }
        
        # Add the face with specific ID
        data['encodings'].append(face_encoding)
        data['ids'].append(face_id)
        data['names'][face_id] = name
        data['counter'] = max(data['counter'], face_id + 1)
        data['saved_at'] = datetime.now().isoformat()
        
        # Save database
        with open('face_database.pkl', 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✅ Added face: ID {face_id} -> {name}")
        return True
        
    except Exception as e:
        print(f"❌ Error adding face: {e}")
        return False

def capture_face_from_camera(face_id, name):
    """Capture a face from camera and add it with specific ID and name"""
    print(f"📷 Starting camera to capture face for ID {face_id} ({name})")
    print("Press SPACE to capture, ESC to cancel")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB for face detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        # Draw rectangles around faces
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {face_id}: {name}", (left, top-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(frame, "SPACE: Capture, ESC: Cancel", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Capture Face', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            print("❌ Capture cancelled")
            cap.release()
            cv2.destroyAllWindows()
            return False
        elif key == 32:  # SPACE key
            if len(face_locations) == 0:
                print("❌ No face detected, try again")
                continue
            
            # Save the captured frame and process it
            cv2.imwrite('temp_capture.jpg', frame)
            cap.release()
            cv2.destroyAllWindows()
            
            # Process the captured image
            success = add_face_with_id_and_name('temp_capture.jpg', face_id, name)
            
            # Clean up temp file
            if os.path.exists('temp_capture.jpg'):
                os.remove('temp_capture.jpg')
            
            return success
    
    cap.release()
    cv2.destroyAllWindows()
    return False

def setup_taha_and_salah():
    """Set up the database with Taha (ID 1) and Salah (ID 2)"""
    print("🎯 Setting up face database for Taha (ID 1) and Salah (ID 2)")
    print("=" * 60)
    
    # Reset database first
    reset_database()
    
    # Capture Taha (ID 1)
    print("\n👤 Setting up Taha (ID 1)")
    if capture_face_from_camera(1, "Taha"):
        print("✅ Taha added successfully!")
    else:
        print("❌ Failed to add Taha")
        return
    
    input("\nPress Enter when ready to capture Salah...")
    
    # Capture Salah (ID 2)  
    print("\n👤 Setting up Salah (ID 2)")
    if capture_face_from_camera(2, "Salah"):
        print("✅ Salah added successfully!")
    else:
        print("❌ Failed to add Salah")
        return
    
    print("\n🎉 Database setup complete!")
    show_database_status()

def show_database_status():
    """Show current database status"""
    try:
        if os.path.exists('face_database.pkl'):
            with open('face_database.pkl', 'rb') as f:
                data = pickle.load(f)
            
            print("\n📊 Current Database Status:")
            print("-" * 40)
            print(f"Total faces: {len(data['encodings'])}")
            print(f"Face IDs: {data['ids']}")
            print(f"Names: {data['names']}")
            print(f"Next ID: {data['counter']}")
            print(f"Last saved: {data['saved_at']}")
        else:
            print("❌ No database found")
    except Exception as e:
        print(f"❌ Error reading database: {e}")

def main():
    """Main menu"""
    while True:
        print("\n" + "=" * 60)
        print("MANUAL FACE SETUP TOOL")
        print("=" * 60)
        print("1. Setup Taha (ID 1) and Salah (ID 2) from camera")
        print("2. Add face from image file")
        print("3. Add face from camera")
        print("4. Reset database")
        print("5. Show database status")
        print("6. Exit")
        print("-" * 60)
        
        try:
            choice = input("Enter your choice (1-6): ").strip()
            
            if choice == '1':
                setup_taha_and_salah()
            
            elif choice == '2':
                image_path = input("Enter image path: ").strip()
                if not os.path.exists(image_path):
                    print("❌ Image file not found")
                    continue
                    
                try:
                    face_id = int(input("Enter face ID: "))
                    name = input("Enter name: ").strip()
                    if name:
                        add_face_with_id_and_name(image_path, face_id, name)
                    else:
                        print("❌ Name cannot be empty")
                except ValueError:
                    print("❌ Invalid face ID")
            
            elif choice == '3':
                try:
                    face_id = int(input("Enter face ID: "))
                    name = input("Enter name: ").strip()
                    if name:
                        capture_face_from_camera(face_id, name)
                    else:
                        print("❌ Name cannot be empty")
                except ValueError:
                    print("❌ Invalid face ID")
            
            elif choice == '4':
                confirm = input("Are you sure you want to reset the database? (yes/no): ")
                if confirm.lower() == 'yes':
                    reset_database()
                else:
                    print("❌ Reset cancelled")
            
            elif choice == '5':
                show_database_status()
            
            elif choice == '6':
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()