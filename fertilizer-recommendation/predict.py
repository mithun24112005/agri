import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RF_MODEL_PATH = os.path.join(BASE_DIR, "rf_pipeline.joblib")
FERTNAME_DICT_PATH = os.path.join(BASE_DIR, "fertname_dict.joblib")
CROPTYPE_DICT_PATH = os.path.join(BASE_DIR, "croptype_dict.joblib")
SOILTYPE_DICT_PATH = os.path.join(BASE_DIR, "soiltype_dict.joblib")

SOIL_TYPES = ["Black", "Clayey", "Loamy", "Red", "Sandy"]
CROP_TYPES = ["Barley", "Cotton", "Ground Nuts", "Maize", "Millets",
              "Oil seeds", "Paddy", "Pulses", "Sugarcane", "Tobacco", "Wheat"]


def predict_fertilizer(temperature, humidity, moisture, soil_type, crop_type,
                       nitrogen, potassium, phosphorous):
    model = joblib.load(RF_MODEL_PATH)
    fertname_dict = joblib.load(FERTNAME_DICT_PATH)

    soil_type = soil_type.strip().title()
    crop_type = crop_type.strip().title()

    if soil_type not in SOIL_TYPES:
        raise ValueError(f"Invalid soil type '{soil_type}'. Choose from: {', '.join(SOIL_TYPES)}")
    if crop_type not in CROP_TYPES:
        raise ValueError(f"Invalid crop type '{crop_type}'. Choose from: {', '.join(CROP_TYPES)}")

    soil_encoded = SOIL_TYPES.index(soil_type)
    crop_encoded = CROP_TYPES.index(crop_type)

    data = pd.DataFrame([{
        "Temparature": temperature,
        "Humidity ": humidity,
        "Moisture": moisture,
        "Soil Type": soil_encoded,
        "Crop Type": crop_encoded,
        "Nitrogen": nitrogen,
        "Potassium": potassium,
        "Phosphorous": phosphorous,
    }])

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred_encoded = int(model.predict(data)[0])
    fertilizer = fertname_dict[pred_encoded]

    return fertilizer


if __name__ == "__main__":
    print("  Fertilizer Recommendation System")
    print()

    print(f"Soil Types: {', '.join(SOIL_TYPES)}")
    print(f"Crop Types: {', '.join(CROP_TYPES)}")
    print()

    temperature = float(input("Temperature (deg C)  : "))
    humidity = float(input("Humidity (%)         : "))
    moisture = float(input("Moisture (%)         : "))
    soil_type = input("Soil Type            : ")
    crop_type = input("Crop Type            : ")
    nitrogen = float(input("Nitrogen             : "))
    potassium = float(input("Potassium            : "))
    phosphorous = float(input("Phosphorous          : "))

    fertilizer = predict_fertilizer(temperature, humidity, moisture,
                                    soil_type, crop_type,
                                    nitrogen, potassium, phosphorous)

    print()
    print(f"  Recommended Fertilizer: {fertilizer}")
