import os
import numpy as np
import joblib
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

from feature_extraction import extract_features_from_path

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}

MODEL_PATH = os.path.join(BASE_DIR, "BrainTumor_Model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "Scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "LabelEncoder.pkl")

app = Flask(__name__)
app.secret_key = "change-this-secret-key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB max upload

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------------------------------------
# Load model artifacts once at startup
# ---------------------------------------------------------
print("Loading model artifacts...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)

# Defensive handling: if the pkl file turns out to hold a dict of models
# (e.g. saved via joblib.dump(models, ...) instead of joblib.dump(best_model, ...)),
# pick a single usable estimator out of it instead of crashing at predict time.
if isinstance(model, dict):
    print(f"Loaded object is a dict with keys: {list(model.keys())}")
    if "SVM" in model:
        model = model["SVM"]
    else:
        # fall back to the first model in the dict
        first_key = next(iter(model))
        print(f"'SVM' key not found, falling back to '{first_key}'")
        model = model[first_key]

print("Model, scaler, and label encoder loaded successfully.")
print("Using model type:", type(model).__name__)
print("Classes:", list(encoder.classes_))


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_brain_tumor(image_path):
    """Run the full feature extraction -> scale -> predict pipeline on one image."""
    features = extract_features_from_path(image_path)
    features = features.reshape(1, -1)
    features = scaler.transform(features)

    prediction = model.predict(features)
    predicted_class = encoder.inverse_transform(prediction)[0]

    confidence = None
    class_probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence = float(np.max(probabilities) * 100)
        class_probabilities = {
            cls: round(float(prob) * 100, 2)
            for cls, prob in zip(encoder.classes_, probabilities)
        }

    return predicted_class, confidence, class_probabilities


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        flash("No file part in the request.")
        return redirect(url_for("index"))

    file = request.files["image"]

    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Unsupported file type. Please upload a PNG, JPG, JPEG, BMP, or TIFF image.")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        predicted_class, confidence, class_probabilities = predict_brain_tumor(filepath)
    except Exception as e:
        flash(f"Error processing image: {e}")
        return redirect(url_for("index"))

    image_url = url_for("static", filename=f"uploads/{filename}")

    return render_template(
        "index.html",
        prediction=predicted_class,
        confidence=confidence,
        class_probabilities=class_probabilities,
        image_url=image_url,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)