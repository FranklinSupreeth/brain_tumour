# Brain Tumor MRI Classifier — Flask App

Serves your trained SVM model (HOG + LBP + GLCM features) through a simple
upload-and-predict web UI.

## 1. Folder structure

```
brain_tumor_app/
├── app.py
├── feature_extraction.py
├── requirements.txt
├── Best_BrainTumor_Model.pkl   <- copy from your notebook output
├── Scaler.pkl                  <- copy from your notebook output
├── LabelEncoder.pkl            <- copy from your notebook output
├── templates/
│   └── index.html
└── static/
    └── uploads/                <- uploaded images get saved here
```

**Copy your three `.pkl` files** (`Best_BrainTumor_Model.pkl`, `Scaler.pkl`,
`LabelEncoder.pkl` — produced in cell 33 of your notebook) into the same
folder as `app.py`.

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Run

```bash
python app.py
```

Open http://localhost:5000 in your browser, upload an MRI image, and get the
predicted class + confidence + per-class probability breakdown.

## How it works

`feature_extraction.py` exactly mirrors the preprocessing from your notebook,
which is critical — the scaler and model were fit on this specific feature
representation, so any drift here will silently break predictions:

1. Read image → resize to 128×128 → convert to grayscale
2. Extract HOG features (9 orientations, 8×8 cells, 2×2 blocks)
3. Extract LBP features (uniform, radius=2, 16 points) as a normalized histogram
4. Extract GLCM texture features (contrast, dissimilarity, homogeneity, energy, correlation, ASM)
5. Concatenate all three into one feature vector
6. Scale with the saved `StandardScaler`
7. Predict with the saved SVM (`predict` + `predict_proba`)
8. Decode the numeric label back to a class name with the saved `LabelEncoder`

## Notes

- Your notebook trained on 4 classes based on the `Testing/glioma` folder in
  your validation code — actual class names are read dynamically from
  `LabelEncoder.pkl` at startup, so whatever classes you trained on will work
  automatically (no hardcoding).
- `opencv-python-headless` is used instead of `opencv-python` since this is a
  server app with no GUI — avoids unnecessary system dependencies.
- Max upload size is capped at 8MB (`MAX_CONTENT_LENGTH` in `app.py`) — raise
  this if your MRI files are larger.
- For production, don't use Flask's dev server (`debug=True`) — run behind
  `gunicorn`/`waitress` instead, and set a real `app.secret_key`.
