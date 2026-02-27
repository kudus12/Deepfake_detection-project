# app.py
# Week 6: Flask app connects to trained model and predicts Real/Fake from uploaded video

import os
import uuid
from flask import Flask, render_template, request

from ML.inference import load_my_model, predict_video  # keep this

app = Flask(__name__)

# -----------------------------
# 1) Upload folder
# -----------------------------
UPLOAD_FOLDER = "uploads"
DEBUG_FRAMES_FOLDER = "debug_frames"  # where we save the face crops used
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DEBUG_FRAMES_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -----------------------------
# 2) Load model once
# -----------------------------
MODEL_PATH = "saved_models/best_model.pth"
my_model = load_my_model(MODEL_PATH)

# -----------------------------
# 3) Home
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# -----------------------------
# Helper: safe extension check
# -----------------------------
def allowed_video(filename: str) -> bool:
    filename = filename.lower()
    return filename.endswith(".mp4") or filename.endswith(".mov") or filename.endswith(".avi") or filename.endswith(".mkv")


# -----------------------------
# 4) Predict
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", error="No file uploaded.")

    file = request.files["file"]

    if file.filename == "":
        return render_template("index.html", error="No file selected.")

    if not allowed_video(file.filename):
        return render_template("index.html", error="Please upload a video file (.mp4/.mov/.avi/.mkv).")

    # ✅ IMPORTANT: save with a unique name to avoid overwriting / caching issues
    ext = os.path.splitext(file.filename)[1].lower()
    unique_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)

    file.save(save_path)

    # ✅ Make a unique debug folder per upload
    per_video_debug = os.path.join(DEBUG_FRAMES_FOLDER, os.path.splitext(unique_name)[0])
    os.makedirs(per_video_debug, exist_ok=True)

    # ✅ Prediction (uses debug_dir to save what the model actually sees)
    label, confidence, probs = predict_video(
        my_model,
        save_path,
        frame_limit=16,
        flip_labels=False,
        debug_dir=per_video_debug
    )

    if label == "ERROR":
        return render_template("index.html", error="Could not read this video. Try another one.")

    return render_template(
        "index.html",
        prediction=label,
        confidence=f"{confidence:.2f}",
        prob_real=f"{probs[0]*100:.2f}",
        prob_fake=f"{probs[1]*100:.2f}",
        filename=file.filename,          # original filename for display
        saved_as=unique_name             # optional if you want to show it
    )


if __name__ == "__main__":
    app.run(debug=True)