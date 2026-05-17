import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "crop_recommendation_rf_model.pkl")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "target_encoder.pkl")

PH_CATEGORY_MAP = {"Acidic": 0, "Alkaline": 1, "Neutral": 2}
RAINFALL_LEVEL_MAP = {"High": 0, "Low": 1, "Medium": 2, "Very High": 3}


def classify_ph(ph):
    if ph < 5.5:
        return "Acidic"
    elif ph <= 7.5:
        return "Neutral"
    else:
        return "Alkaline"


def classify_rainfall(rainfall):
    if rainfall <= 50:
        return "Low"
    elif rainfall <= 100:
        return "Medium"
    elif rainfall <= 200:
        return "High"
    else:
        return "Very High"


def predict_crop(n, p, k, temperature, humidity, ph, rainfall):
    model = joblib.load(MODEL_PATH)
    target_encoder = joblib.load(ENCODER_PATH)

    ph_cat = classify_ph(ph)
    rain_level = classify_rainfall(rainfall)

    data = pd.DataFrame([{
        "N": n,
        "P": p,
        "K": k,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall,
        "NPK_mean": (n + p + k) / 3,
        "THI": (temperature * humidity) / 100,
        "ph_category": PH_CATEGORY_MAP[ph_cat],
        "rainfall_level": RAINFALL_LEVEL_MAP[rain_level],
    }])

    pred_encoded = model.predict(data)[0]
    crop = target_encoder.inverse_transform([pred_encoded])[0]

    return crop


if __name__ == "__main__":
    print("=" * 50)
    print("  Crop Recommendation System")
    print("=" * 50)
    print()

    n = float(input("Nitrogen (N)         : "))
    p = float(input("Phosphorus (P)       : "))
    k = float(input("Potassium (K)        : "))
    temperature = float(input("Temperature (deg C)  : "))
    humidity = float(input("Humidity (%)         : "))
    ph = float(input("pH of soil           : "))
    rainfall = float(input("Rainfall (mm)        : "))

    crop = predict_crop(n, p, k, temperature, humidity, ph, rainfall)

    print()
    print("=" * 50)
    print(f"  Recommended Crop: {crop}")
    print("=" * 50)
