from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Folder to store uploads
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename: str) -> bool:
    return "." in filename and "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def dummy_deepfake_detector(filename: str):
    """
    DEMO LOGIC:

    - If filename is exactly "pic1.jpg" / "pic1.png" / "pic1.jpeg" -> Clean / Real
    - If filename is exactly "pic2.jpg" / "pic2.png" / "pic2.jpeg" -> Not clean / Deepfake
    - Otherwise -> Unknown (demo only)
    """
    name = filename.lower()

    pic1_names = {"pic1.jpg", "pic1.png", "pic1.jpeg"}
    pic2_names = {"pic2.jpg", "pic2.png", "pic2.jpeg"}

    if name in pic1_names:
        label = "Clean / Real image"
        status = "real"
        confidence = 92.5
    elif name in pic2_names:
        label = "Not clean / Deepfake detected"
        status = "fake"
        confidence = 89.1
    else:
        label = "Unknown image (demo logic only)"
        status = "unknown"
        confidence = 50.0

    return label, status, confidence


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    status = None
    confidence = None
    filename = None
    error = None

    if request.method == "POST":
        if "file" not in request.files:
            error = "No file part in the request."
            return render_template("index.html", error=error)

        file = request.files["file"]

        if file.filename == "":
            error = "No file selected."
            return render_template("index.html", error=error)

        if file and allowed_file(file.filename):
            safe_name = secure_filename(file.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
            file.save(save_path)

            prediction, status, confidence = dummy_deepfake_detector(safe_name)
            filename = safe_name

            return render_template(
                "index.html",
                prediction=prediction,
                status=status,
                confidence=round(confidence, 2),
                filename=filename,
                error=None,
            )
        else:
            error = "Unsupported file type. Please upload a JPG or PNG image."

    return render_template(
        "index.html",
        prediction=prediction,
        status=status,
        confidence=confidence,
        filename=filename,
        error=error,
    )


# Route to serve uploaded images so they can be displayed in the page
@app.route("/uploads/<path:filename>")
def uploaded_files(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True)
