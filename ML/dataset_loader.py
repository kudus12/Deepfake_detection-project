import os
import cv2
import torch
from torch.utils.data import Dataset

class DeepfakeVideoDataset(Dataset):
    def __init__(self, real_dir, fake_dir, frame_limit=10):
        self.samples = []
        self.frame_limit = frame_limit

        # Label: 0 = real, 1 = fake
        for file in os.listdir(real_dir):
            if file.endswith(".mp4"):
                self.samples.append((os.path.join(real_dir, file), 0))

        for file in os.listdir(fake_dir):
            if file.endswith(".mp4"):
                self.samples.append((os.path.join(fake_dir, file), 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
     video_path, label = self.samples[index]
     cap = cv2.VideoCapture(video_path)

     frames = []
     count = 0

     while cap.isOpened() and count < self.frame_limit:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (224, 224))
        frame = frame.astype("float32") / 255.0
        frame = torch.from_numpy(frame).permute(2, 0, 1)  # C,H,W
        frames.append(frame)
        count += 1

     cap.release()
 
    # IMPORTANT: avoid crashing if a video fails to decode
     if len(frames) == 0:
        frames = torch.zeros((self.frame_limit, 3, 224, 224), dtype=torch.float32)
     else:
        # pad if video shorter than frame_limit
        while len(frames) < self.frame_limit:
            frames.append(frames[-1].clone())
        frames = torch.stack(frames[:self.frame_limit])

     return frames, label

if __name__ == "__main__":
    REAL_DIR = r"C:\Users\kudus\OneDrive - Atlantic TU\4th year\Research in Computing\full-progress-code\Dataset\real\original"
    FAKE_DIR = r"C:\Users\kudus\OneDrive - Atlantic TU\4th year\Research in Computing\full-progress-code\Dataset\fake\Deepfakes"

    dataset = DeepfakeVideoDataset(REAL_DIR, FAKE_DIR)

    print("Total videos loaded:", len(dataset))

    frames, label = dataset[0]
    print("Frames shape:", frames.shape)
    print("Label (0=real, 1=fake):", label)
