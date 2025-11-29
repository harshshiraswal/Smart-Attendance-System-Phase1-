"""
Face Detection and Recognition Module
Phase 1 - Smart Attendance System
"""

import face_recognition
import os
import sys
import cv2
import numpy as np
import time

# Import config without cv2 dependencies
from config import *

class FaceRecognitionSystem:
    """
    Main class for face detection and recognition system
    """
    
    def __init__(self):
        # Initialize face tracking variables
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.known_face_encodings = []
        self.known_face_names = []
        self.process_current_frame = True
        self.previously_recognized = []
        self.frame_count = 0
        
        # Setup directories and encode known faces
        self.setup_directories()
        self.encode_known_faces()
        
        print("=" * 50)
        print("Face Recognition System Initialized Successfully!")
        print(f"Loaded {len(self.known_face_names)} known faces")
        print("=" * 50)

    def setup_directories(self):
        """Create necessary directories if they don't exist"""
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        print(f"Directories verified:")
        print(f"- Known faces: {KNOWN_FACES_DIR}")
        print(f"- Logs: {LOG_DIR}")

    def encode_known_faces(self):
        """Encode all known faces from the known_faces directory"""
        try:
            images = [f for f in os.listdir(KNOWN_FACES_DIR) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not images:
                print("⚠️  No face images found in known_faces directory")
                return

            for image_file in images:
                image_path = os.path.join(KNOWN_FACES_DIR, image_file)
                
                # Load and encode face
                face_image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(face_image)
                
                if face_encodings:
                    self.known_face_encodings.append(face_encodings[0])
                    name = os.path.splitext(image_file)[0]
                    self.known_face_names.append(name)
                    print(f"✓ Encoded face: {name}")
                else:
                    print(f"⚠️  No face found in: {image_file}")
            
        except Exception as e:
            print(f"❌ Error encoding known faces: {e}")

    def calculate_face_confidence(self, face_distance, face_match_threshold=0.6):
        """Calculate confidence percentage for face matches"""
        range_val = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range_val * 2.0)

        if face_distance > face_match_threshold:
            return str(round(linear_val * 100, 2)) + '%'
        else:
            value = (linear_val + ((1.0 - linear_val) * 
                     pow((linear_val - 0.5) * 2, 0.2))) * 100
            return str(round(value, 2)) + '%'

    def process_frame(self, frame):
        """Process a single frame for face detection and recognition"""
        small_frame = cv2.resize(frame, (0, 0), fx=FRAME_SCALE_FACTOR, 
                                fy=FRAME_SCALE_FACTOR)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find all faces in current frame
        self.face_locations = face_recognition.face_locations(rgb_small_frame)
        self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
        
        self.face_names = []
        recognized_names = []

        for face_encoding in self.face_encodings:
            # Compare with known faces
            matches = face_recognition.compare_faces(
                self.known_face_encodings, face_encoding, tolerance=FACE_MATCH_THRESHOLD)
            face_distances = face_recognition.face_distance(
                self.known_face_encodings, face_encoding)
            
            best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else 0
            name = "Unknown"
            confidence = "??%"
            
            if len(face_distances) > 0:
                confidence = self.calculate_face_confidence(
                    face_distances[best_match_index], FACE_MATCH_THRESHOLD)
                confidence_value = float(confidence[:-1])

                if confidence_value > CONFIDENCE_THRESHOLD and matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    recognized_names.append(name)

            display_name = f'{name} ({confidence})'
            self.face_names.append(display_name)

        return recognized_names

    def draw_face_annotations(self, frame, scale_factor=4):
        """Draw bounding boxes and names on detected faces"""
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            # Scale coordinates back to original frame size
            top = int(top * scale_factor)
            right = int(right * scale_factor)
            bottom = int(bottom * scale_factor)
            left = int(left * scale_factor)

            # Draw bounding box around face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            
            # Draw background rectangle for name
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), 
                         (0, 0, 255), cv2.FILLED)
            
            # Draw name and confidence text
            cv2.putText(frame, name, (left + 6, bottom - 6), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

    def run_recognition(self):
        """Main method to run face recognition system"""
        print("🚀 Starting face recognition system...")
        print("📷 Initializing camera...")
        
        # Use default camera without Windows-specific API
        video_capture = cv2.VideoCapture(CAMERA_INDEX)

        if not video_capture.isOpened():
            print("❌ Error: Could not access camera")
            sys.exit(1)

        print("✅ Camera initialized successfully")
        print("🎯 Press 'q' to quit the application")
        print("-" * 50)

        start_time = time.time()
        frame_count = 0

        try:
            while True:
                ret, frame = video_capture.read()
                
                if not ret:
                    print("❌ Error: Could not read frame from camera")
                    break

                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process every other frame to improve performance
                self.frame_count += 1
                if self.frame_count % PROCESS_EVERY_N_FRAME == 0:
                    recognized_names = self.process_frame(frame)
                    self.draw_face_annotations(frame, int(1/FRAME_SCALE_FACTOR))
                    self.process_current_frame = True
                else:
                    self.process_current_frame = False

                # Calculate and display FPS
                frame_count += 1
                fps = frame_count / (time.time() - start_time)
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Display the resulting frame
                cv2.imshow(DISPLAY_WINDOW_NAME, frame)

                # Break loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⏹️  Stopping face recognition system...")
                    break

        except KeyboardInterrupt:
            print("\n⏹️  System interrupted by user")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
        finally:
            # Clean up resources
            video_capture.release()
            cv2.destroyAllWindows()
            print("✅ Camera resources released")
            print("🎉 Face recognition system stopped successfully.")


if __name__ == '__main__':
    try:
        face_system = FaceRecognitionSystem()
        face_system.run_recognition()
    except Exception as e:
        print(f"❌ Failed to start face recognition system: {e}")
