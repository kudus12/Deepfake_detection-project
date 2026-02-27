# ML/inference.py
# Week 6: Inference (prediction) code for Flask
# Goal: load saved model + predict Real/Fake from uploaded video
#
# IMPORTANT: This matches TRAINING preprocessing:
# - face crop (haar cascade)
# - BGR -> RGB
# - normalize /255
# - sequential frames from start (up to frame_limit)
#
# If you want, you can enable debug to save the face crops used.

import os
import cv2
import torch
import torch.nn.functional as F

from ML.model import SimpleCNN  # IMPORTANT: ML.model (not just model)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Load model
# -----------------------------
def load_my_model(model_path: str):
    """
    Loads trained CNN weights from .pth file and returns model in eval mode.
    """
    model = SimpleCNN().to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)

    # If you ever saved a checkpoint dict:
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.eval()
    return model


# -----------------------------
# Face crop (MATCH TRAINING)
# -----------------------------
def _make_face_detector():
    haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(haar_path)


def _crop_face_like_training(frame, face_detector, face_size=224):
    """
    Same as dataset_loader.py:
    - detect biggest face
    - if no face, center square crop
    - resize to face_size
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        crop = frame[y:y + h, x:x + w]
        did_detect = True
    else:
        h, w, _ = frame.shape
        s = min(h, w)
        cx, cy = w // 2, h // 2
        x1 = max(cx - s // 2, 0)
        y1 = max(cy - s // 2, 0)
        crop = frame[y1:y1 + s, x1:x1 + s]
        did_detect = False

    crop = cv2.resize(crop, (face_size, face_size))
    return crop, did_detect


# -----------------------------
# Video -> Tensor (MATCH TRAINING exactly)
# -----------------------------
def video_to_tensor(video_path: str, frame_limit=16, size=224, debug_dir=None):
    """
    Returns tensor shape: (1, T, C, H, W)
    MATCH TRAINING:
    - sequential frames from start until frame_limit
    - face crop
    - BGR -> RGB
    - /255
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None

    face_detector = _make_face_detector()

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    frames = []
    count = 0
    face_hits = 0

    while cap.isOpened() and count < frame_limit:
        ret, frame = cap.read()
        if not ret:
            break

        face_bgr, did_detect = _crop_face_like_training(frame, face_detector, face_size=size)
        if did_detect:
            face_hits += 1

        # OPTIONAL: save crops the model sees (BGR)
        if debug_dir:
            tag = "FACE" if did_detect else "CENTER"
            cv2.imwrite(os.path.join(debug_dir, f"{count:02d}_{tag}.jpg"), face_bgr)

        # MATCH TRAINING: BGR -> RGB
        face_rgb = face_bgr[:, :, ::-1]

        # normalize
        face_rgb = face_rgb.astype("float32") / 255.0

        # to tensor (C,H,W)
        face_tensor = torch.from_numpy(face_rgb).permute(2, 0, 1)

        frames.append(face_tensor)
        count += 1

    cap.release()

    if len(frames) == 0:
        return None

    # Pad if short (MATCH TRAINING)
    while len(frames) < frame_limit:
        frames.append(frames[-1].clone())

    frames = torch.stack(frames[:frame_limit])  # (T,C,H,W)
    frames = frames.unsqueeze(0)                # (1,T,C,H,W)

    # Helpful print (you can remove later)
    if debug_dir:
        print(f"[DEBUG] Face detected in {face_hits}/{frame_limit} frames for {os.path.basename(video_path)}")

    return frames


# -----------------------------
# Predict
# -----------------------------
def predict_video(model, video_path: str, frame_limit=16, flip_labels=False, debug_dir=None):
    """
    Returns:
      label (str): "REAL" or "FAKE"
      confidence (float): 0..100
      probs (list): [prob_real, prob_fake]
    """
    frames = video_to_tensor(video_path, frame_limit=frame_limit, size=224, debug_dir=debug_dir)
    if frames is None:
        return "ERROR", 0.0, [0.0, 0.0]

    frames = frames.to(DEVICE)

    with torch.no_grad():
        logits = model(frames)              # (1,2)
        probs = F.softmax(logits, dim=1)    # (1,2)

        prob_real = float(probs[0, 0].item())  # 0 = REAL (your training)
        prob_fake = float(probs[0, 1].item())  # 1 = FAKE

        pred_idx = int(torch.argmax(probs, dim=1).item())

        if flip_labels:
            pred_idx = 1 - pred_idx
            prob_real, prob_fake = prob_fake, prob_real

        if pred_idx == 0:
            return "REAL", prob_real * 100, [prob_real, prob_fake]
        else:
            return "FAKE", prob_fake * 100, [prob_real, prob_fake]