import os
import cv2

# Path to real videos dataset
REAL_DIR = r"C:\Users\kudus\OneDrive - Atlantic TU\4th year\Research in Computing\full-progress-code\real\original"

# Path to fake (deepfake) videos dataset
FAKE_DIR = r"C:\Users\kudus\OneDrive - Atlantic TU\4th year\Research in Computing\full-progress-code\fake\Deepfakes"

# Function to test if videos can be opened and read
def test_videos(folder, limit=5):
    # Get first few mp4 videos from the folder
    vids = [f for f in os.listdir(folder) if f.lower().endswith(".mp4")][:limit]

    # If no videos are found, print message and stop
    if not vids:
        print("No mp4 videos found in:", folder)
        return

    # Loop through each video and test it
    for v in vids:
        path = os.path.join(folder, v)
        cap = cv2.VideoCapture(path)  # Open video file
        ok, frame = cap.read()        # Try to read the first frame
        opened = cap.isOpened()       # Check if video opened correctly
        cap.release()                 # Close the video file

        # Print results for each video
        print("\nTesting:", path)
        print("Opened:", opened)
        print("First frame read:", ok)

        # If frame is read successfully, print frame details
        if ok:
            print("Frame shape:", frame.shape)
            print("Video decoded successfully ✅")
        else:
            print("Failed to decode video ❌")

# Test real videos
print("=== REAL VIDEOS ===")
test_videos(REAL_DIR)

# Test fake videos
print("\n=== FAKE VIDEOS ===")
test_videos(FAKE_DIR)
