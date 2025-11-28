# Smart Attendance System - Phase 1: Face Detection Module

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-orange)
![Face Recognition](https://img.shields.io/badge/Face--Recognition-1.3-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📋 Project Overview
This is **Phase 1** of the **Smart Attendance System Using Facial Recognition** project developed during my Vocational Training at **OLF**. This module implements real-time face detection and recognition capabilities.

## 🎯 Features
- **Real-time Face Detection** using OpenCV
- **Face Recognition** with confidence scoring
- **Automatic Attendance Logging** with timestamps
- **Multiple Face Support** in single frame
- **External Webcam Compatibility**
- **Attendance Duplication Prevention**

## 🏗️ Project Structure
Smart-Attendance-System-Phase1/
├── main.py # Main application entry point
├── face_detection.py # Core face detection class
├── config.py # Configuration settings
├── requirements.txt # Project dependencies
├── utils/
│ └── helpers.py # Utility functions
├── known_faces/ # Directory for known face images
├── logs/ # Attendance logs directory
└── samples/
└── sample_usage.py # Example usage script

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Webcam (internal or external)

# 📁 File Descriptions

main.py: Application entry point with command-line interface
face_detection.py: Core face recognition logic and camera handling
config.py: All configurable parameters (thresholds, paths, camera settings)
utils/helpers.py: Utility functions for confidence calculation and logging
samples/sample_usage.py: Example script for testing

# 🔧 Configuration

Modify config.py to adjust:

FACE_MATCH_THRESHOLD: Face matching sensitivity (0.6 recommended)
CONFIDENCE_THRESHOLD: Minimum confidence percentage (97% recommended)
CAMERA_INDEX: Camera device index (0 for default camera)
File paths and logging preferences

# 📊 Output

Real-time Display: Bounding boxes with names and confidence percentages
Attendance Logs: Timestamped entries in logs/attendance_log.txt
Console Feedback: Recognition events and system status

# 🛠️ Technologies Used

Python - Primary programming language
OpenCV - Computer vision and camera handling
face_recognition - Face detection and encoding
NumPy - Numerical computations

# 📝 Development Context

This project was developed as part of my Vocational Training at OLF and represents Phase 1 of a comprehensive Smart Attendance System. The training focused on practical implementation of computer vision and AI concepts in real-world applications.

# 🔮 Next Phase (Phase 2)

Phase 2 will extend this system with:

Database integration for persistent storage
Web-based administration interface
Advanced reporting and analytics
Multi-camera support
Mobile application integration

# 🤝 Contributing

This is a personal project developed during vocational training. While primarily for demonstration purposes, suggestions and feedback are welcome!

# 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

# 🚀 Usage

Allow camera access when prompted

Face the camera - detected faces will show with names and confidence scores

Recognized faces are automatically logged in logs/attendance_log.txt

Press 'q' to quit the application

Run python main.py
```bash
python main.py

