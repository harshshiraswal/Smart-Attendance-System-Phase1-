"""
Smart Attendance System - Phase 1: Face Detection Module
Developed during Vocational Training at OLF
Enhanced Professional Version with Stable Face Recognition
"""

import cv2
import face_recognition
import os
import numpy as np
import time
import argparse
import sys
from datetime import datetime


class ProfessionalFaceRecognition:
    def __init__(self, camera_index=0):
        self.known_face_encodings = []
        self.known_face_names = []
        self.camera_index = camera_index
        self.previous_face_data = {}  # Store previous frame data for stability
        self.load_known_faces()
        
    def display_banner(self):
        """Display professional application banner"""
        banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║         SMART ATTENDANCE SYSTEM - PHASE 1                    ║
    ║         Professional Face Detection & Recognition            ║
    ║                                                              ║
    ║         Developed during Vocational Training at OLF          ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)

    def load_known_faces(self):
        """Load and encode known faces with professional logging"""
        print("🎯 SMART ATTENDANCE SYSTEM - PHASE 1")
        print("=" * 60)
        print("🏢 Developed during Vocational Training at OLF")
        print("📊 Loading known faces database...")
        print("-" * 60)
        
        known_faces_dir = "known_faces"
        loaded_count = 0
        
        # Create directories if they don't exist
        os.makedirs(known_faces_dir, exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # Check if known_faces directory has images
        if not os.path.exists(known_faces_dir):
            print("❌ CRITICAL: known_faces directory not found!")
            return
        
        image_files = [f for f in os.listdir(known_faces_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        if not image_files:
            print("❌ No face images found in known_faces directory!")
            print("💡 Please add face images as: known_faces/person_name.jpg")
            return
        
        for image_file in image_files:
            image_path = os.path.join(known_faces_dir, image_file)
            try:
                # Load image and find face encodings
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                
                if encodings:
                    self.known_face_encodings.append(encodings[0])
                    name = os.path.splitext(image_file)[0].replace('_', ' ').title()
                    self.known_face_names.append(name)
                    print(f"✅ {name}")
                    loaded_count += 1
                else:
                    print(f"❌ No face detected in: {image_file}")
                    
            except Exception as e:
                print(f"⚠️  Error loading {image_file}: {e}")
        
        print("-" * 60)
        print(f"🎯 Database Ready: {loaded_count} faces loaded")
        print("=" * 60)
        
        if loaded_count == 0:
            print("❌ CRITICAL: No valid faces found in database")
            return False
        return True

    def calculate_face_confidence(self, face_distance, face_match_threshold=0.6):
        """
        Calculate confidence percentage for face matches
        This is the standard formula used in face recognition systems
        """
        if face_distance > face_match_threshold:
            range_val = (1.0 - face_match_threshold)
            linear_val = (1.0 - face_distance) / (range_val * 2.0)
            return linear_val * 100
        else:
            range_val = (1.0 - face_match_threshold)
            linear_val = (1.0 - face_distance) / (range_val * 2.0)
            value = (linear_val + ((1.0 - linear_val) * np.power((linear_val - 0.5) * 2, 0.2))) * 100
            return value

    def draw_stable_bbox(self, frame, face_id, left, top, right, bottom, name, confidence):
        """Draw stable bounding box that only updates text, not the entire box"""
        # Generate a consistent color based on face_id for stability
        color_seed = hash(face_id) % 5
        colors = [(0, 255, 0), (0, 255, 255), (255, 255, 0), (255, 165, 0), (0, 191, 255)]
        color = colors[color_seed]
        
        # Store current face data for stability
        current_face_key = f"{left}_{top}"
        
        # Use previous position if available for smooth transitions
        if current_face_key in self.previous_face_data:
            prev_data = self.previous_face_data[current_face_key]
            # Smooth position transition
            left = int(0.7 * prev_data['left'] + 0.3 * left)
            top = int(0.7 * prev_data['top'] + 0.3 * top)
            right = int(0.7 * prev_data['right'] + 0.3 * right)
            bottom = int(0.7 * prev_data['bottom'] + 0.3 * bottom)
        
        # Update stored data
        self.previous_face_data[current_face_key] = {
            'left': left, 'top': top, 'right': right, 'bottom': bottom,
            'name': name, 'confidence': confidence
        }
        
        # Draw main bounding box (consistent appearance)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
        
        # Draw corner markers (consistent)
        corner_length = 12
        thickness = 2
        
        # Top-left corner
        cv2.line(frame, (left, top), (left + corner_length, top), color, thickness)
        cv2.line(frame, (left, top), (left, top + corner_length), color, thickness)
        
        # Top-right corner
        cv2.line(frame, (right, top), (right - corner_length, top), color, thickness)
        cv2.line(frame, (right, top), (right, top + corner_length), color, thickness)
        
        # Bottom-left corner
        cv2.line(frame, (left, bottom), (left + corner_length, bottom), color, thickness)
        cv2.line(frame, (left, bottom), (left, bottom - corner_length), color, thickness)
        
        # Bottom-right corner
        cv2.line(frame, (right, bottom), (right - corner_length, bottom), color, thickness)
        cv2.line(frame, (right, bottom), (right, bottom - corner_length), color, thickness)
        
        # Draw name background (static size to prevent flickering)
        label_bg_height = 35
        label_bg_top = top - label_bg_height
        label_bg_bottom = top
        label_bg_left = left
        label_bg_right = left + 200  # Fixed width to prevent resizing
        
        # Draw background
        cv2.rectangle(frame, (label_bg_left, label_bg_top), 
                     (label_bg_right, label_bg_bottom), color, -1)
        
        # Prepare text (only this changes)
        if name == "Unknown":
            label = f"UNKNOWN {confidence:.1f}%"
            text_color = (255, 255, 255)
        else:
            label = f"{name} {confidence:.1f}%"
            text_color = (0, 0, 0)  # Black text for better readability
        
        # Add text with shadow for better readability
        cv2.putText(frame, label, (left + 8, top - 12), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 2)  # Shadow
        cv2.putText(frame, label, (left + 8, top - 12), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, text_color, 1)  # Main text

    def display_system_info(self, frame, fps, faces_detected, recognized_faces):
        """Display professional system information overlay"""
        # Main header background
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        
        # System title
        title = "Smart Attendance System - Phase 1 | OLF Vocational Training"
        cv2.putText(frame, title, (10, 25), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
        
        # Status information on right side
        status_text = f"FPS: {fps:.1f} | Faces: {faces_detected} | Recognized: {recognized_faces}"
        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.putText(frame, status_text, (frame.shape[1] - text_size[0] - 10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Footer with instructions
        footer_bg = frame.shape[0] - 30
        cv2.rectangle(frame, (0, footer_bg), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        instruction = "Press 'Q' to exit system | Real-time Face Recognition Active"
        cv2.putText(frame, instruction, (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def test_camera_access(self):
        """Test camera access before starting main loop"""
        print("📷 Testing camera access...")
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"✅ Camera {i} is working!")
                    cap.release()
                    return i
                cap.release()
            print(f"❌ Camera {i} not accessible")
        
        print("❌ No working camera found!")
        return None

    def improve_recognition_accuracy(self, face_encoding):
        """Enhanced recognition with better matching logic"""
        if not self.known_face_encodings:
            return "Unknown", 0.0
        
        # Calculate face distances
        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
        
        # Find the best match
        best_match_index = np.argmin(face_distances)
        best_distance = face_distances[best_match_index]
        
        # Calculate confidence using standard formula
        confidence = self.calculate_face_confidence(best_distance)
        
        # Apply recognition threshold
        if confidence > 85:  # 85% confidence threshold
            return self.known_face_names[best_match_index], confidence
        else:
            return "Unknown", confidence

    def run_recognition(self, demo_mode=False):
        """Main recognition loop with stable face tracking"""
        print("🚀 Starting Real-time Face Recognition System")
        
        # Test camera first
        working_camera = self.test_camera_access()
        if working_camera is None:
            print("💡 Camera Troubleshooting:")
            print("1. Check if camera is connected")
            print("2. Grant camera permissions to Terminal")
            print("3. Close other apps using camera")
            print("4. Try different camera index: python main.py --camera 1")
            return
        
        # Use the working camera index
        self.camera_index = working_camera
        
        print("📷 Initializing camera...")
        
        # Initialize camera with optimal settings
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("❌ Camera initialization failed")
            return
        
        print("✅ Camera ready")
        print("🎬 Starting recognition loop...")
        print("=" * 60)
        print("💡 Face the camera and ensure good lighting")
        print("🎯 System will identify you in real-time")
        print("=" * 60)
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Frame capture failed")
                    break
                
                # Mirror the frame for more natural interaction
                frame = cv2.flip(frame, 1)
                
                # Initialize variables for this frame
                face_locations = []
                face_names = []
                face_confidences = []
                recognized_count = 0
                
                # Process every other frame for performance
                frame_count += 1
                if frame_count % 2 == 0:
                    # Resize for faster processing while maintaining quality
                    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    
                    # Detect face locations and encodings
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                    
                    for i, face_encoding in enumerate(face_encodings):
                        # Use enhanced recognition
                        name, confidence = self.improve_recognition_accuracy(face_encoding)
                        
                        face_names.append(name)
                        face_confidences.append(confidence)
                        
                        if name != "Unknown":
                            recognized_count += 1
                    
                    # Draw stable bounding boxes
                    for (top, right, bottom, left), name, confidence in zip(face_locations, face_names, face_confidences):
                        # Scale back up face locations
                        top *= 2; right *= 2; bottom *= 2; left *= 2
                        
                        # Create unique face ID for consistent tracking
                        face_id = f"face_{left}_{top}"
                        
                        self.draw_stable_bbox(frame, face_id, left, top, right, bottom, name, confidence)
                
                # Clean up old face data to prevent memory leaks
                if frame_count % 50 == 0:
                    self.previous_face_data.clear()
                
                # Calculate FPS
                current_time = time.time()
                fps = frame_count / (current_time - start_time) if (current_time - start_time) > 0 else 0
                
                # Display professional system overlay
                self.display_system_info(frame, fps, len(face_locations), recognized_count)
                
                # Show frame
                cv2.imshow('Smart Attendance System - Phase 1 | OLF Vocational Training', frame)
                
                # Exit on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("⏹️  System shutdown initiated...")
                    break
                    
        except KeyboardInterrupt:
            print("⏹️  System interrupted by user")
        except Exception as e:
            print(f"❌ System error: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Camera resources released")
            print("🎉 System shutdown complete")
            print("=" * 60)


def main():
    """
    Main function to run the enhanced face recognition system
    """
    parser = argparse.ArgumentParser(
        description='Smart Attendance System - Phase 1: Professional Face Detection Module'
    )
    parser.add_argument('--demo', action='store_true',
                       help='Run in demo mode (for testing)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index (0 for default, 1 for external)')
    
    args = parser.parse_args()
    
    # Display professional banner
    system = ProfessionalFaceRecognition(camera_index=args.camera)
    system.display_banner()
    
    print("🚀 Initializing Smart Attendance System...")
    print("📍 Phase 1: Professional Face Detection Module")
    print("🏢 Developed during Vocational Training at OLF")
    print("=" * 60)
    
    if args.demo:
        print("🔧 Running in DEMO mode...")
    
    if args.camera != 0:
        print(f"📷 Using camera index: {args.camera}")
    
    try:
        # Initialize and run enhanced face recognition system
        print("🔄 Starting enhanced face recognition system...")
        system.run_recognition(demo_mode=args.demo)
        
    except KeyboardInterrupt:
        print("\n⏹️  Application stopped by user.")
    except Exception as e:
        print(f"❌ Error running application: {e}")
        sys.exit(1)
    finally:
        print("\n👋 Thank you for using Smart Attendance System!")
        print("📍 Phase 1 - Professional Face Detection Complete")
        print("🎯 Ready for Phase 2: Database Integration & Web Interface")


if __name__ == '__main__':
    main()