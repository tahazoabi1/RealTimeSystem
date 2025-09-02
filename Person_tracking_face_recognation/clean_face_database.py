#!/usr/bin/env python3
"""
Face Database Cleaner
Keeps only "Salah" and "Taha" in the face database and removes all other entries
"""

import pickle
import os
from face_recognizer import FaceRecognizer

def clean_face_database():
    """Clean face database to keep only Salah and Taha"""
    
    # Initialize face recognizer
    face_recognizer = FaceRecognizer()
    
    print(f"Current face database status:")
    print(f"   Total faces: {len(face_recognizer.known_face_encodings)}")
    print(f"   Face names: {face_recognizer.face_names}")
    
    # Find Salah and Taha in the current database
    salah_data = None
    taha_data = None
    
    # Look for Salah and Taha by name
    for face_id, name in face_recognizer.face_names.items():
        if name.lower() == "salah":
            # Find index of this face_id
            if face_id in face_recognizer.known_face_ids:
                idx = face_recognizer.known_face_ids.index(face_id)
                salah_data = {
                    'id': face_id,
                    'encoding': face_recognizer.known_face_encodings[idx],
                    'name': name
                }
                print(f"Found Salah with ID {face_id}")
        elif name.lower() == "taha":
            # Find index of this face_id
            if face_id in face_recognizer.known_face_ids:
                idx = face_recognizer.known_face_ids.index(face_id)
                taha_data = {
                    'id': face_id,
                    'encoding': face_recognizer.known_face_encodings[idx],
                    'name': name
                }
                print(f"Found Taha with ID {face_id}")
    
    # Check what we found
    if salah_data is None:
        print("WARNING: Salah not found in database")
    if taha_data is None:
        print("WARNING: Taha not found in database")
    
    if salah_data is None and taha_data is None:
        print("ERROR: Neither Salah nor Taha found in database!")
        print("You need to add them first using the face recognition system")
        return False
    
    # Clear the database
    face_recognizer.known_face_encodings = []
    face_recognizer.known_face_ids = []
    face_recognizer.face_names = {}
    face_recognizer.face_id_counter = 1
    
    print("Cleared existing database")
    
    # Add back only Salah and Taha
    if salah_data:
        face_recognizer.known_face_encodings.append(salah_data['encoding'])
        face_recognizer.known_face_ids.append(salah_data['id'])
        face_recognizer.face_names[salah_data['id']] = salah_data['name']
        face_recognizer.face_id_counter = max(face_recognizer.face_id_counter, salah_data['id'] + 1)
        print(f"Re-added Salah with ID {salah_data['id']}")
    
    if taha_data:
        face_recognizer.known_face_encodings.append(taha_data['encoding'])
        face_recognizer.known_face_ids.append(taha_data['id'])
        face_recognizer.face_names[taha_data['id']] = taha_data['name']
        face_recognizer.face_id_counter = max(face_recognizer.face_id_counter, taha_data['id'] + 1)
        print(f"Re-added Taha with ID {taha_data['id']}")
    
    # Save the cleaned database
    face_recognizer.save_face_database()
    
    print("Saved cleaned face database")
    print(f"Final database status:")
    print(f"   Total faces: {len(face_recognizer.known_face_encodings)}")
    print(f"   Face names: {face_recognizer.face_names}")
    print(f"   Next ID counter: {face_recognizer.face_id_counter}")
    
    return True

def list_current_faces():
    """List all current faces in the database"""
    face_recognizer = FaceRecognizer()
    
    print("\nCurrent Face Database:")
    print("=" * 40)
    
    if len(face_recognizer.known_face_encodings) == 0:
        print("   (Empty database)")
        return
    
    face_list = face_recognizer.list_faces_with_names()
    for face in face_list:
        print(f"   ID {face['id']}: {face['name']}")

def main():
    """Main function"""
    print("Face Database Cleaner")
    print("=" * 40)
    
    # Show current database
    list_current_faces()
    
    # Ask for confirmation
    print("\nWARNING: This will remove ALL faces except 'Salah' and 'Taha'")
    response = input("Continue? (y/N): ").strip().lower()
    
    if response == 'y' or response == 'yes':
        success = clean_face_database()
        if success:
            print("\nDatabase cleaned successfully!")
            list_current_faces()
        else:
            print("\nDatabase cleaning failed!")
    else:
        print("Operation cancelled")

if __name__ == "__main__":
    main()