# AGRI — Smart Agriculture Assistant

An all-in-one Python project that helps farmers make data-driven decisions using machine learning, deep learning, and live government data. It covers four core areas: **crop recommendation**, **fertilizer recommendation**, **plant disease detection**, and **mandi (market) pricing**.

---

## Features at a Glance

| Feature | What it does | How it works |
|---|---|---|
| Crop Recommendation | Tells you the best crop to grow on your land | You enter soil nutrients (N, P, K), temperature, humidity, pH, and rainfall. A trained Random Forest model picks the best crop from 22 options. |
| Fertilizer Recommendation | Tells you which fertilizer to use for your crop | You enter temperature, humidity, moisture, soil type, crop type, nitrogen, potassium, and phosphorus. A trained Random Forest pipeline picks the right fertilizer from 7 options. |
| Disease Detection | Identifies plant diseases from a leaf photo | You upload a photo of a plant leaf. An EfficientNet-B4 deep learning model (trained on 38 disease classes) tells you what disease the plant has. |
| Mandi Pricing | Shows live market prices for crops | You enter a crop name and optional location. The script fetches real-time wholesale prices (min, max, modal) from the Indian government's data.gov.in API. |

---

## Project Structure

```
AGRI/
│
├── .venv/                          # Shared Python virtual environment (uv-managed)
├── requirements.txt                # Python packages needed for the project
├── .gitignore                      # Files to ignore in git
├── .env                            # API keys (data.gov.in key for mandi pricing)
│
├── crop-recommendation/            # Feature 1: Crop Recommendation
│   ├── Crop_recommendation.csv     # Training dataset (22 crops, 2200 records)
│   ├── crop_recommendation_rf_model.pkl   # Trained Random Forest model
│   ├── target_encoder.pkl          # Label encoder (converts numbers back to crop names)
│   ├── crop-recommendation-system.ipynb   # Notebook used to train the model
│   ├── predict.py                  # Simple script to get crop recommendations
│   └── examples.txt                # Example inputs and expected outputs
│
├── fertilizer-recommendation/      # Feature 2: Fertilizer Recommendation
│   ├── Fertilizer Prediction.csv   # Training dataset (99 records, 7 fertilizers)
│   ├── rf_pipeline.joblib          # Trained Random Forest pipeline (with StandardScaler)
│   ├── svm_pipeline.joblib         # Trained SVM pipeline (alternative model)
│   ├── xgb_pipeline.joblib         # Trained XGBoost pipeline (alternative model)
│   ├── croptype_dict.joblib        # Crop name → number mapping
│   ├── soiltype_dict.joblib        # Soil type → number mapping
│   ├── fertname_dict.joblib        # Fertilizer name → number mapping
│   ├── fertilizersrecommendation.ipynb    # Notebook used to train the models
│   ├── predict.py                  # Simple script to get fertilizer recommendations
│   └── examples.txt                # Example inputs and expected outputs
│
├── api/                            # Feature 3: Disease Detection API (FastAPI)
│   ├── main.py                     # FastAPI app entry point — starts the server
│   ├── config.py                   # Settings: image size, model path, allowed file types
│   │
│   ├── routes/
│   │   └── disease_routes.py       # API endpoint: POST /disease/predict
│   │
│   ├── services/
│   │   └── disease_service.py      # Full prediction pipeline (validate → save → preprocess → predict → clean up)
│   │
│   ├── utils/
│   │   ├── image_utils.py          # Validates image format and preprocesses it for the model
│   │   └── prediction_utils.py     # Converts model output into readable disease names + confidence scores
│   │
│   ├── models/
│   │   ├── disease_model.py        # Loads the EfficientNet-B4 model once at startup (singleton pattern)
│   │   └── pretrained/
│   │       └── plant_disease_efficientnetb4.h5   # Pretrained deep learning model (from HuggingFace)
│   │
│   ├── data/
│   │   └── labels.py               # All 38 PlantVillage disease class names
│   │
│   └── uploads/                    # Temporary folder for uploaded images (cleaned after prediction)
│
├── disease-detection/              # Notebook for disease detection exploration
│   └── plant-disease-detection.ipynb
│
├── mandi_pricing/                  # Feature 4: Mandi (Market) Pricing
│   └── mandi.py                    # Script to fetch live crop prices from data.gov.in
│
└── README.md                       # This file
```

---

## How Each Feature Works

### 1. Crop Recommendation

**What you give it:**
- Nitrogen (N), Phosphorus (P), Potassium (K) levels in your soil
- Temperature in °C
- Humidity in %
- Soil pH value
- Rainfall in mm

**What it does:**
1. Creates 4 extra features from your input (average NPK, temperature-humidity index, pH category, rainfall level)
2. Scales everything using MinMaxScaler
3. Runs it through a Random Forest model trained on 2,200 records covering 22 crops
4. Returns the best crop to grow

**Accuracy:** ~99%

### 2. Fertilizer Recommendation

**What you give it:**
- Temperature, humidity, moisture
- Soil type (Black, Clayey, Loamy, Red, Sandy)
- Crop type (11 options like Maize, Cotton, Wheat, etc.)
- Nitrogen, Potassium, Phosphorus values

**What it does:**
1. Converts soil type and crop type into numbers
2. Scales all values using StandardScaler
3. Runs it through a Random Forest pipeline
4. Returns the best fertilizer from 7 options (Urea, DAP, 14-35-14, 28-28, 17-17-17, 20-20, 10-26-26)

**Accuracy:** 100% on training data

### 3. Disease Detection (API)

**What you give it:**
- A photo of a plant leaf (JPG, JPEG, or PNG)

**What it does:**
1. Checks that the file is a valid image
2. Resizes it to 380x380 pixels
3. Runs it through an EfficientNet-B4 deep learning model
4. Returns the disease name, confidence score, and top 3 predictions

**Model:** EfficientNet-B4 trained on PlantVillage dataset (38 disease classes across Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato)

### 4. Mandi Pricing

**What you give it:**
- A crop name (e.g., Tomato, Wheat, Rice)
- Optional: state, district, or market name

**What it does:**
1. Calls the Indian government's data.gov.in API
2. Fetches the latest wholesale market prices
3. Shows min price, max price, and modal (most common) price in ₹ per quintal

---

## How to Set Up and Run

### Step 1 — Activate the virtual environment

```powershell
.venv\Scripts\activate
```

### Step 2 — Install dependencies (if not already done)

```powershell
uv pip install -r requirements.txt
```

### Step 3 — Test Crop Recommendation

```powershell
python crop-recommendation\predict.py
```

Enter the 7 values when prompted. Example:
```
Nitrogen (N)         : 90
Phosphorus (P)       : 42
Potassium (K)        : 43
Temperature (deg C)  : 20.88
Humidity (%)         : 82.0
pH of soil           : 6.5
Rainfall (mm)        : 202.9
```
Output: `Recommended Crop: rice`

See `crop-recommendation/examples.txt` for more examples.

### Step 4 — Test Fertilizer Recommendation

```powershell
python fertilizer-recommendation\predict.py
```

Enter the 8 values when prompted. Example:
```
Temperature (deg C)  : 26
Humidity (%)         : 52
Moisture (%)         : 38
Soil Type            : Sandy
Crop Type            : Maize
Nitrogen             : 37
Potassium            : 0
Phosphorous          : 0
```
Output: `Recommended Fertilizer: Urea`

See `fertilizer-recommendation/examples.txt` for more examples.

### Step 5 — Test Disease Detection API

Start the server:
```powershell
$env:TF_USE_LEGACY_KERAS="1"; uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Wait for `Model loaded successfully.` then:

- **Swagger docs:** Open `http://127.0.0.1:8000/docs` in your browser
- **Health check:** `GET http://127.0.0.1:8000/health`
- **Predict disease:** `POST http://127.0.0.1:8000/disease/predict` (upload a leaf image as form-data with key `file`)

**cURL example:**
```bash
curl -X POST "http://127.0.0.1:8000/disease/predict" -F "file=@leaf_image.jpg"
```

**Expected response:**
```json
{
    "disease": "Tomato___Late_blight",
    "confidence": 0.9843,
    "top_predictions": [
        {"label": "Tomato___Late_blight", "confidence": 0.9843},
        {"label": "Tomato___Early_blight", "confidence": 0.012},
        {"label": "Tomato___Leaf_Mold", "confidence": 0.003}
    ]
}
```

### Step 6 — Test Mandi Pricing

```powershell
python mandi_pricing\mandi.py
```

Enter a crop name and optional location filters. It will show live wholesale prices from Indian mandis.

---

## Quick Reference

| Feature | Run command | Input type | Output |
|---|---|---|---|
| Crop Recommendation | `python crop-recommendation\predict.py` | 7 numeric values | Best crop name |
| Fertilizer Recommendation | `python fertilizer-recommendation\predict.py` | 6 numeric + 2 text values | Best fertilizer name |
| Disease Detection | `uvicorn api.main:app --reload` | Leaf image (JPG/PNG) | Disease name + confidence |
| Mandi Pricing | `python mandi_pricing\mandi.py` | Crop name + optional location | Live market prices |
