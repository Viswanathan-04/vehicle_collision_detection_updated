import cv2
import os

def extract_frames(video_path, output_folder, num_frames=50):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return
    
    # Get total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate the interval to pick frames
    frame_interval = total_frames // num_frames
    
    count = 0
    extracted_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if count % frame_interval == 0:
            # Save frame as image
            frame_filename = os.path.join(output_folder, f"C_14_{extracted_frames+1:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            extracted_frames += 1
            
            # Stop if we've extracted enough frames
            if extracted_frames >= num_frames:
                break
        
        count += 1
    
    cap.release()
    print(f"Extracted {extracted_frames} frames to {output_folder}")

# Example usage
video_path = './C_01_001 (2).mp4'
output_folder = 'extracted_frames'
extract_frames(video_path, "./Dataset_new")
