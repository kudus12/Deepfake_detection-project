# dataset_loader.py
# My dataset loader: reads videos and returns FACE-cropped frames (T,C,H,W)

import os
import cv2
import torch
from torch.utils.data import Dataset

class DeepfakeVideoDataset(Dataset):
    def __init__(self, real_dir, fake_dir, frame_limit=16, face_size=224):
        self.samples = []
        self.frame_limit = frame_limit
        self.face_size = face_size

        # My face detector (simple + works for student projects)
        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_detector = cv2.CascadeClassifier(haar_path)

        for file in os.listdir(real_dir):
            if file.lower().endswith(".mp4"):
                self.samples.append((os.path.join(real_dir, file), 0))

        for file in os.listdir(fake_dir):
            if file.lower().endswith(".mp4"):
                self.samples.append((os.path.join(fake_dir, file), 1))

    def __len__(self):
        return len(self.samples)

    def _crop_face(self, frame):
        # My helper: detect biggest face. If not found, do center crop.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            crop = frame[y:y+h, x:x+w]
        else:
            h, w, _ = frame.shape
            s = min(h, w)
            cx, cy = w // 2, h // 2
            x1 = max(cx - s // 2, 0)
            y1 = max(cy - s // 2, 0)
            crop = frame[y1:y1+s, x1:x1+s]

        crop = cv2.resize(crop, (self.face_size, self.face_size))
        return crop

    def __getitem__(self, index):
        video_path, label = self.samples[index]
        cap = cv2.VideoCapture(video_path)

        frames = []
        count = 0

        while cap.isOpened() and count < self.frame_limit:
            ret, frame = cap.read()
            if not ret:
                break

            face = self._crop_face(frame)

            # BGR -> RGB
            face = face[:, :, ::-1]

            # Normalize
            face = face.astype("float32") / 255.0

            # To tensor (C,H,W)
            face = torch.from_numpy(face).permute(2, 0, 1)

            frames.append(face)
            count += 1

        cap.release()

        if len(frames) == 0:
            # My fallback: try next video
            return self.__getitem__((index + 1) % len(self.samples))

        while len(frames) < self.frame_limit:
            frames.append(frames[-1].clone())

        frames = torch.stack(frames[:self.frame_limit])  # (T,C,H,W)
        return frames, label
