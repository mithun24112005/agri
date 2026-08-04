# AGRI — Architecture Document

System architecture, data flows, module dependencies, and design decisions for the Smart Agriculture Assistant.

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AGRI System                              │
│                                                                 │
│   ┌──────────────────────┐   ┌──────────────────────────────┐  │
│   │  CLI Scripts          │   │  FastAPI Web Server          │  │
│   │                       │   │                              │  │
│   │  crop-recommendation/ │   │  api/                        │  │
│   │    predict.py         │   │    main.py  ← entry point    │  │
│   │                       │   │    routes/   ← endpoints     │  │
│   │  fertilizer-recommend/│   │    services/  ← business     │  │
│   │    predict.py         │   │    utils/     ← helpers      │  │
│   │                       │   │    models/    ← ML model     │  │
│   │  mandi_pricing/       │   │                              │  │
│   │    mandi.py           │   └──────────────────────────────┘  │
│   │                       │                                     │
│   └──────────────────────┘                                     │
│                                                                 │
│   ┌──────────────────────────────────────────────────────┐      │
│   │  Shared Resources                                     │      │
│   │  ┌─────────┐ ┌──────────────┐ ┌──────────────────┐   │      │
│   │  │ .venv   │ │requirements  │ │ .env (API keys)  │   │      │
│   │  │(Python) │ │ (packages)   │ │                  │   │      │
│   │  └─────────┘ └──────────────┘ └──────────────────┘   │      │
│   └──────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

### Two access modes

| Mode | Interface | Typical user | Latency |
|---|---|---|---|
| **CLI** | Command-line prompts | Farmer with terminal access | Seconds (local inference) |
| **API** | HTTP POST (upload image) | App developer integrating into a mobile/web app | ~1-2s (model inference) |
| **CLI (Mandi)** | Command-line prompts | Farmer checking live prices | ~2-5s (network API call) |

---

## 2. Module Dependency Graph

```
┌──────────────────────────────────────────────────────────────────┐
│  crop-recommendation/predict.py                                  │
│     ├── joblib ──────────► crop_recommendation_rf_model.pkl     │
│     ├── joblib ──────────► target_encoder.pkl                   │
│     └── pandas (for DataFrame input shaping)                    │
│                                                                  │
│  fertilizer-recommendation/predict.py                            │
│     ├── joblib ──────────► rf_pipeline.joblib                   │
│     ├── joblib ──────────► fertname_dict.joblib                 │
│     └── pandas (for DataFrame input shaping)                    │
│                                                                  │
│  mandi_pricing/mandi.py                                          │
│     ├── subprocess ──────► curl.exe (external binary)            │
│     ├── dotenv ───────────► .env (DATA_GOV_API_KEY)             │
│     └── data.gov.in API (external HTTPS)                        │
│                                                                  │
│  api/main.py (FastAPI)                                           │
│     ├── api/models/disease_model.py                              │
│     │    └── tensorflow ──► models/pretrained/*.h5              │
│     ├── api/routes/disease_routes.py                             │
│     │    └── api/services/disease_service.py                     │
│     │         ├── api/utils/image_utils.py                       │
│     │         │    └── PIL, numpy, tensorflow.preprocess_input  │
│     │         ├── api/utils/prediction_utils.py                  │
│     │         │    └── api/data/labels.py                       │
│     │         └── api/models/disease_model.py                    │
│     └── api/config.py (constants used by all layers)            │
└──────────────────────────────────────────────────────────────────┘
```

**Key rule:** All modules import downward (config → utils → services → routes → main). No circular imports.

---

## 3. Component Architectures

### 3.1 Crop Recommendation — Data Flow

```
        User Input                  Feature Engineering              Inference
  ┌──────────────────────┐    ┌─────────────────────────────┐    ┌──────────────┐
  │  N, P, K             │    │  NPK_mean = (N+P+K)/3       │    │ Random Forest│
  │  Temperature         │───►│  THI = temp * humidity / 100 │───►│  Model       │
  │  Humidity            │    │  ph_category (Acidic/        │    │  (.pkl)      │
  │  pH                  │    │    Neutral/Alkaline → 0,1,2) │    └──────┬───────┘
  │  Rainfall            │    │  rainfall_level (Low/Medium/ │           │
  └──────────────────────┘    │    High/Very High → 0,1,2,3) │           ▼
                              └─────────────────────────────┘    ┌──────────────┐
                                                                │ 22 crops     │
                                                                │ (LabelEncoder│
                                                                │  inverse)    │
                                                                └──────────────┘
```

**Model details:**
- Algorithm: Random Forest Classifier
- Training data: 2,200 records, 22 classes (100 samples per crop)
- Input features (after engineering): 7 raw → 10 engineered
- Preprocessing: MinMaxScaler (embedded in pickled `Pipeline`)
- Feature importance: N, P, K, temperature, humidity, pH, rainfall, NPK_mean, THI, ph_category, rainfall_level

### 3.2 Fertilizer Recommendation — Data Flow

```
        User Input               Encoding + Scaling             Inference
  ┌──────────────────────┐    ┌─────────────────────────┐    ┌──────────────┐
  │  Temperature         │    │  Soil Type (text→index) │    │ Random Forest│
  │  Humidity            │───►│  Crop Type (text→index) │───►│  Pipeline    │
  │  Moisture            │    │  StandardScaler on all  │    │  (.joblib)   │
  │  Soil Type           │    │    numeric columns      │    └──────┬───────┘
  │  Crop Type           │    └─────────────────────────┘           │
  │  N, K, P             │                                          ▼
  └──────────────────────┘                             ┌────────────────────┐
                                                       │  fertname_dict     │
                                                       │  (index → name)    │
                                                       │  7 fertilizers     │
                                                       └────────────────────┘
```

**Model details:**
- Algorithm: Random Forest Classifier wrapped in `Pipeline(StandardScaler, RandomForestClassifier)`
- Training data: 99 records, 7 fertilizer classes
- Validation: SVM and XGBoost pipelines also saved, RF chosen for 100% accuracy
- Categorical encoding: manual index-based (hardcoded order), not LabelEncoder

### 3.3 Disease Detection API — Architecture

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│  FastAPI Server (uvicorn)                                                         │
│                                                                                   │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐  │
│  │  Client  │──►│  main.py     │──►│  routes/     │──►│  services/           │  │
│  │  (HTTP)  │   │  lifespan    │   │  /disease/   │   │  disease_service.py  │  │
│  │          │   │  load model  │   │  predict     │   │                      │  │
│  └──────────┘   └──────────────┘   └──────────────┘   └──────┬───────────────┘  │
│                                                               │                  │
│                   ┌───────────────────────────────────────────┼──────────────┐   │
│                   │  Inside predict_disease()                 │              │   │
│                   │                                           ▼              │   │
│                   │  ┌──────────────┐   ┌──────────────┐   ┌──────────┐    │   │
│                   │  │ utils/       │   │ utils/       │   │ models/  │    │   │
│                   │  │ image_utils  │   │ pred_utils   │   │ disease_ │    │   │
│                   │  │ validate +   │   │ decode +     │   │ model    │    │   │
│                   │  │ preprocess   │   │ top-k        │   │ .predict │    │   │
│                   │  └──────┬───────┘   └──────┬────────┘   └────┬─────┘    │   │
│                   │         │                  │                │          │   │
│                   │         └──────────────────┴────────────────┘          │   │
│                   │                         Result                         │   │
│                   └─────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────────────────────┘
```

**Request lifecycle:**
```
POST /disease/predict (multipart/form-data: file=<image>)
    │
    ▼
disease_routes.predict(file: UploadFile)
    │
    ▼
disease_service.predict_disease(file)
    │
    ├── 1. validate_image(filename)
    │         └── Check extension is jpg/jpeg/png
    │
    ├── 2. Save image to disk (uuid-based unique name)
    │
    ├── 3. preprocess_image(file_path)
    │         ├── PIL.Image.open → convert RGB
    │         ├── resize to (380, 380)
    │         ├── numpy array → expand_dims → add batch dim
    │         └── efficientnet.preprocess_input (normalize)
    │
    ├── 4. disease_model.predict(preprocessed)
    │         └── model.predict() → shape (1, 38) probabilities
    │
    ├── 5. decode_prediction(predictions)
    │         ├── argmax → class index
    │         └── CLASS_NAMES[index] → disease name
    │
    ├── 6. get_top_predictions(predictions, top_k=3)
    │         └── argsort descending → top 3 labels + confidences
    │
    └── 7. Delete uploaded file from disk
    │
    ▼
Return JSON:
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

**Model details:**
- Architecture: EfficientNet-B4 (pretrained on ImageNet, fine-tuned on PlantVillage)
- Input size: 380×380×3 (RGB)
- Output: 38-class softmax
- Framework: TensorFlow 2.x with legacy Keras (`TF_USE_LEGACY_KERAS=1`)
- Model source: [huggingface.co/liriope/PlantDiseaseDetection](https://huggingface.co/liriope/PlantDiseaseDetection)
- Model weight: ~270 MB (`.h5` format)
- Singleton pattern: `DiseaseModel` class loads once at server startup (FastAPI lifespan), not per-request

### 3.4 Mandi Pricing — Data Flow

```
        User Input                           Subprocess Call                 External API
  ┌──────────────────────┐              ┌──────────────────────┐       ┌─────────────────────┐
  │  Commodity (required)│              │  subprocess.run()    │       │  data.gov.in         │
  │  State (optional)    │─────────────►│  curl -s --ssl-revoke│──────►│  /resource/          │
  │  District (optional) │              │  -A "Mozilla/..."    │       │  9ef84268-d588-...   │
  │  Market (optional)   │              │  + query params      │       │                     │
  └──────────────────────┘              └──────────────────────┘       └─────────┬───────────┘
                                                                                 │
                                                                                 ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  Parse JSON response                                                        │
  │  ┌───────────────────────────────────────────────────────────────────────┐  │
  │  │ [                                                                     │  │
  │  │   {"market":"Yeshwanthpur","district":"Bengaluru","state":"Karnataka",│  │
  │  │    "arrival_date":"2024-01-15","min_price":800,"max_price":2800,      │  │
  │  │    "modal_price":2000}, ...                                           │  │
  │  │ ]                                                                     │  │
  │  └───────────────────────────────────────────────────────────────────────┘  │
  │                                                                             │
  │  Formatted table output:                                                    │
  │  Market              District       State       Min    Max    Modal         │
  │  ──────────────────────────────────────────────────────────────────────     │
  │  Yeshwanthpur        Bengaluru      Karnataka    800   2800   2000          │
  │  Hubli               Dharwad        Karnataka    600   1800   1200          │
  └─────────────────────────────────────────────────────────────────────────────┘
```

**Why `subprocess` + `curl` instead of `requests`:**
- Windows SSL issues with `requests` + data.gov.in (SSL certificate verification fails in some environments)
- `curl` with `--ssl-no-revoke` bypasses Windows cert revocation checks
- User-agent and Referer headers mimic a browser to avoid API blocking

---

## 4. Design Decisions

### 4.1 Legacy Keras Compatibility

TensorFlow 2.21 ships Keras 3 as default. The PlantVillage EfficientNet-B4 model was saved under Keras 2's `TFOpLambda`ops. Loading it without legacy mode throws:
```
ValueError: Unrecognized keyword arguments: ... TFOpLambda
```

**Solution:** `os.environ["TF_USE_LEGACY_KERAS"] = "1"` must be set **before** `import tensorflow`. This is done redundantly in:
- `api/main.py:3`
- `api/models/disease_model.py:3`
- `api/utils/image_utils.py:3`

This ensures robustness — even if one module is imported first, legacy Keras is active.

### 4.2 Singleton Model Pattern

```python
class DiseaseModel:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self): ...
    def predict(self, image): ...

disease_model = DiseaseModel()
```

- `DiseaseModel()` returns the same instance every time (singleton via `__new__`)
- `load()` is called once in FastAPI's `lifespan` context — when the server starts
- `predict()` auto-loads on first call as a fallback
- This avoids re-loading the 270 MB model on every request

### 4.3 Upload Cleanup

Images uploaded via the API are deleted immediately after prediction (`os.remove(file_path)`) to prevent disk bloat. The UUID-prefixed filenames prevent collisions under concurrent uploads.

### 4.4 Model File Management

All `.pkl` / `.joblib` / `.h5` model files are stored inside their respective feature directories, **not** git-tracked:
- `.gitignore` covers `.venv/`, `uploads/`, `__pycache__/`
- `.h5` model is downloaded separately (not committed)
- Only notebooks, scripts, and config are version-controlled

### 4.5 API Key Handling

The data.gov.in API key is stored in `.env` (gitignored) and loaded via `python-dotenv`. A default key is hardcoded as fallback in `mandi.py` for convenience.

---

## 5. API Endpoints

| Method | Path | Description | Request | Response |
|---|---|---|---|---|
| `GET` | `/` | Root health check | — | `{"message": "AGRI API Running"}` |
| `GET` | `/health` | Server health | — | `{"status": "healthy"}` |
| `POST` | `/disease/predict` | Predict plant disease | `multipart/form-data` with `file` field (JPG/PNG) | `{disease, confidence, top_predictions}` |

Swagger UI: `http://127.0.0.1:8000/docs`

---

## 6. Data Sources

| Feature | Data source | Records | Format |
|---|---|---|---|
| Crop Recommendation | Kaggle (Crop Recommendation Dataset) | 2,200 | CSV |
| Fertilizer Recommendation | Kaggle (Fertilizer Prediction Dataset) | 99 | CSV |
| Disease Detection | PlantVillage (via HuggingFace) | 54,306 (38 classes) | H5 model weights |
| Mandi Pricing | data.gov.in (Open Government Data Platform India) | Live | JSON API |

---

## 7. Training Pipelines

Each feature has a corresponding Jupyter notebook that documents the full training process:

| Notebook | Location | Algorithms tried | Best model |
|---|---|---|---|
| `crop-recommendation-system.ipynb` | `crop-recommendation/` | Random Forest, SVM, XGBoost, etc. | Random Forest |
| `fertilizersrecommendation.ipynb` | `fertilizer-recommendation/` | RF Pipeline, SVM Pipeline, XGB Pipeline | RF Pipeline |
| `plant-disease-detection.ipynb` | `disease-detection/` | EfficientNet-B4 fine-tuning | EfficientNet-B4 |

---

## 8. System Requirements

- **Python:** 3.12+
- **OS:** Windows (primary), Linux/macOS (compatible)
- **RAM (min):** 4 GB (8 GB recommended for disease detection model)
- **Disk:** ~1 GB (including model files and venv)
- **Network:** Required for mandi pricing (live API calls) and initial model download
