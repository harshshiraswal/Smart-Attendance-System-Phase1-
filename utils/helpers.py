"""
Utility functions for Face Detection Module
Phase 1 - Smart Attendance System
Developed during Vocational Training at OLF
"""

import math
import datetime
import cv2
import numpy as np
import os


def calculate_face_confidence(face_distance, face_match_threshold=0.6):
    """
    Calculate confidence percentage for face matches
    """
    range_val = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range_val * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * 
                 math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


def setup_directories():
    """Create necessary directories if they don't exist"""
    from config import KNOWN_FACES_DIR, LOG_DIR
    
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    print(f"Directories verified:")
    print(f"- Known faces: {KNOWN_FACES_DIR}")
    print(f"- Logs: {LOG_DIR}")


def log_attendance(name, log_path):
    """
    Log attendance with timestamp to file
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f'{name}, {timestamp}, Attendance marked!\n'
    
    try:
        with open(log_path, 'a') as f:
            f.write(log_entry)
        print(f"✓ Attendance logged for: {name} at {timestamp}")
    except Exception as e:
        print(f"✗ Error logging attendance: {e}")


def draw_face_annotations(frame, face_locations, face_names, scale_factor=4):
    """
    Draw bounding boxes and names on detected faces
    """
    # Use default colors instead of importing from config to avoid circular imports
    bbox_color = (0, 0, 255)  # Red
    text_color = (255, 255, 255)  # White
    
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale coordinates back to original frame size
        top = int(top * scale_factor)
        right = int(right * scale_factor)
        bottom = int(bottom * scale_factor)
        left = int(left * scale_factor)

        # Draw bounding box around face
        cv2.rectangle(frame, (left, top), (right, bottom), bbox_color, 2)
        
        # Draw background rectangle for name
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), 
                     bbox_color, cv2.FILLED)
        
        # Draw name and confidence text
        cv2.putText(frame, name, (left + 6, bottom - 6), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, text_color, 1)
    
    return frame


def validate_known_faces(known_faces_dir):
    """
    Validate known faces directory and images
    """
    valid_extensions = ('.png', '.jpg', '.jpeg')
    image_files = []
    
    if not os.path.exists(known_faces_dir):
        print(f"✗ Known faces directory not found: {known_faces_dir}")
        return image_files
    
    for file in os.listdir(known_faces_dir):
        if file.lower().endswith(valid_extensions):
            image_files.append(file)
    
    print(f"✓ Found {len(image_files)} valid face images")
    return image_files
