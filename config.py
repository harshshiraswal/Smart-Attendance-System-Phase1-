"""
Configuration settings for Face Detection Module
Phase 1 - Smart Attendance System
"""

import os

# Base directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(BASE_DIR, 'known_faces')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'attendance_log.txt')

# Face recognition settings
FACE_MATCH_THRESHOLD = 0.6
CONFIDENCE_THRESHOLD = 97.0
FRAME_SCALE_FACTOR = 0.25

# Camera settings
CAMERA_INDEX = 0  # 0 for default camera

# Display settings (cv2 constants will be imported where used)
DISPLAY_WINDOW_NAME = 'Smart Attendance System - Phase 1'

# Performance settings
PROCESS_EVERY_N_FRAME = 2
