import os
import cv2
import torch
from torch.utils.data import Dataset

# Custom dataset class for loading deepfake and real videos
class DeepfakeVideoDataset(Dataset):
    def __init__(self, real_dir, fake_dir, frame_limit=10):
        # List to store video paths and labels
        self.samples = []
        self.frame_limit = frame_limit  # number of frames per video

        # Label: 0 = real video
        for file in os.listdir(real_dir):
            if file.endswith(".mp4"):
                self.samples.append((os.path.join(real_dir, file), 0))

        # Label: 1 = fake (deepfake) video
        for file in os.listdir(fake_dir):
            if file.endswith(".mp4"):
                self.samples.append((os.path.join(fake_dir, file), 1))

    # Returns total number of videos in dataset
    def __len__(self):
        return len(self.samples)

    # Loads one video and returns frames + label
    def __getitem__(self, index):
        video_path, label = self.samples[index]
        cap = cv2.VideoCapture(video_path)

        frames = []
        count = 0

        # Read frames until frame limit is reached
        while cap.isOpened() and count < self.frame_limit:
            ret, frame = cap.read()
            if not ret:
                break  # stop if video ends or fails

            # Resize frame to model input size
            frame = cv2.resize(frame, (224, 224))

            # Normalize pixel values between 0 and 1
            frame = frame.astype("float32") / 255.0

            # Convert frame to tensor and change format to C,H,W
            frame = torch.from_numpy(frame).permute(2, 0, 1)

            frames.append(frame)
            count += 1

        cap.release()

        # Prevent crash if video decoding fails
        if len(frames) == 0:
            # Create empty frames if video fails
            frames = torch.zeros((self.frame_limit, 3, 224, 224), dtype=torch.float32)
        else:
            # Pad frames if video shorter than frame_limit
            while len(frames) < self.frame_limit:
                frames.append(frames[-1].clone())

            # Stack frames into tensor
            frames = torch.stack(frames[:self.frame_limit])

        # Return video frames and label
        return frames, label


# Test dataset loading when running file directly
if __name__ == "__main__":
    REAL_DIR = r"C:\Users\kudus\OneDrive - Atlantic TU\4th year\Research in Computing\full-progress-code\Dataset\real\original"
    FAKE_DIR = r"C:\Users\kudus\OneDrive - Atlantic TU\4th year\Research in Computing\full-progress-code\Dataset\fake\Deepfakes"

    dataset = DeepfakeVideoDataset(REAL_DIR, FAKE_DIR)

    # Print dataset size
    print("Total videos loaded:", len(dataset))

    # Load one example
    frames, label = dataset[0]
    print("Frames shape:", frames.shape)
    print("Label (0=real, 1=fake):", label)
