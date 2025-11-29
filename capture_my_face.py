import cv2
import os

def capture_face():
    print("📸 Face Capture Utility")
    print("This will save your face to known_faces/ directory")
    
    # Try cameras until we find one that works
    for camera_index in [0, 1, 2]:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            print(f"✅ Found camera at index {camera_index}")
            break
    else:
        print("❌ No camera found. Please check permissions.")
        return
    
    print("\n💡 Instructions:")
    print("- Face the camera directly")
    print("- Ensure good lighting")
    print("- Press SPACEBAR to capture")
    print("- Press ESC to cancel")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame")
            break
            
        # Show instructions on the frame
        cv2.putText(frame, "Press SPACE to capture", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press ESC to cancel", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Capture Your Face', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            print("❌ Capture cancelled")
            break
        elif key == 32:  # SPACE key
            name = input("\nEnter your name (firstname_lastname): ").strip()
            if name:
                filename = f"known_faces/{name}.jpg"
                cv2.imwrite(filename, frame)
                print(f"✅ Photo saved as: {filename}")
                print("🎉 Face added successfully!")
                break
            else:
                print("❌ Please enter a valid name")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_face()
