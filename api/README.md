# AGRI Disease Detection API

This folder contains the FastAPI service used by AGRI to identify plant diseases
from leaf photographs. The service accepts an uploaded image, prepares it for
an EfficientNet-B4 neural network, maps the model's numeric output to the
PlantVillage disease classes represented by the local label list, and returns
the best result together with
the top three predictions.

The API is intentionally focused on disease detection. Crop recommendation,
fertilizer recommendation, and mandi pricing are separate features elsewhere
in the repository.

---

## Contents

- [What the API does](#what-the-api-does)
- [Folder structure](#folder-structure)
- [Application architecture](#application-architecture)
- [Request lifecycle](#request-lifecycle)
- [Files explained](#files-explained)
- [HTTP API](#http-api)
- [Prediction response](#prediction-response)
- [Validation and error behavior](#validation-and-error-behavior)
- [Model and label mapping](#model-and-label-mapping)
- [Configuration reference](#configuration-reference)
- [Local setup and startup](#local-setup-and-startup)
- [Testing the running API](#testing-the-running-api)
- [Operational notes](#operational-notes)
- [Troubleshooting](#troubleshooting)
- [Extension and maintenance guide](#extension-and-maintenance-guide)
- [Current limitations](#current-limitations)

---

## What the API does

### Input

The prediction endpoint expects a multipart/form-data request with one uploaded
file:

| Field | Type | Required | Accepted extensions |
|---|---|---:|---|
| `file` | `UploadFile` | Yes | `.jpg`, `.jpeg`, `.png` |

The implementation uses the file name extension as the first validation step.
The image is then opened with Pillow, converted to RGB, resized to 380 x 380
pixels, and preprocessed with TensorFlow's EfficientNet preprocessing
function.

### Output

The model output is converted to:

1. `disease`: the label with the greatest model score.
2. `confidence`: the greatest score, rounded to four decimal places.
3. `top_predictions`: the three highest-scoring labels and their scores.

The service does not currently return treatment recommendations, crop metadata,
an image URL, or a confidence-threshold status.

---

## Folder structure

The relevant source and runtime layout is:

```text
api/
├── README.md                         # This documentation
├── __init__.py                       # Makes api importable as a package
├── config.py                         # Paths and prediction constants
├── main.py                           # FastAPI application and lifecycle
│
├── data/
│   ├── __init__.py                   # Package marker
│   └── labels.py                     # Ordered list of 37 local output labels
│
├── models/
│   ├── __init__.py                   # Package marker
│   ├── disease_model.py              # Model singleton and inference wrapper
│   └── pretrained/
│       ├── plant_disease_efficientnetb4.h5
│       │                                  # TensorFlow/Keras model artifact
│       └── .cache/                     # Hugging Face download metadata/cache
│
├── routes/
│   ├── __init__.py                   # Package marker
│   └── disease_routes.py             # /disease/predict route
│
├── services/
│   ├── __init__.py                   # Package marker
│   └── disease_service.py            # End-to-end prediction orchestration
│
├── utils/
│   ├── __init__.py                   # Package marker
│   ├── image_utils.py                # File validation and image preprocessing
│   └── prediction_utils.py           # Scores-to-label conversion
│
└── uploads/
    └── <temporary uploaded files>     # Created/used during predictions
```

### Generated and ignored content

The working tree can also contain Python `__pycache__` directories and compiled
`.pyc` files. These are interpreter-generated artifacts and are not part of
the API implementation.

`uploads/` is ignored by the repository's root `.gitignore`. It is a temporary
working directory, not a permanent image store. Normally it should be empty
after successful predictions because the service deletes each uploaded file
after decoding the result.

The `.cache/` directory under `models/pretrained/` contains download metadata
associated with the model artifact. It is not imported by the application.

---

## Application architecture

The API is organized into layers:

```text
HTTP request
    │
    ▼
routes/disease_routes.py
    │  validates route-level invocation and translates unexpected failures
    ▼
services/disease_service.py
    │  coordinates validation, storage, preprocessing, inference, decoding
    ├──────────────► utils/image_utils.py
    │                   └── Pillow + EfficientNet preprocess_input
    ├──────────────► models/disease_model.py
    │                   └── TensorFlow/Keras EfficientNet-B4 artifact
    └──────────────► utils/prediction_utils.py
                            └── data/labels.py
```

`main.py` is the composition root. It creates the FastAPI application, loads
the model during the application lifespan, exposes the root and health routes,
and registers the disease router.

There is no database, authentication layer, background queue, or persistent
prediction record in this folder. Every prediction is computed synchronously
inside the request handling path.

---

## Request lifecycle

### 1. Process startup

When Uvicorn imports `api.main`, the module:

1. Sets `TF_USE_LEGACY_KERAS=1` before TensorFlow-related modules are loaded.
2. Imports the route and singleton model objects.
3. Creates the FastAPI application.
4. Registers the lifespan handler.

When FastAPI starts its lifespan:

1. `disease_model.load()` checks whether the model is already loaded.
2. It prints the TensorFlow version and detected GPU devices.
3. It verifies that `MODEL_PATH` exists.
4. It loads the `.h5` model with `compile=False` and `safe_mode=False`.
5. The server then begins accepting requests.

Model loading is therefore expected to happen once per process at startup. The
model wrapper also has a lazy-load fallback in `predict()` for callers that use
the wrapper outside the normal FastAPI lifespan.

### 2. Route dispatch

`POST /disease/predict` receives an `UploadFile` under the `file` form field
and calls `predict_disease(file)`.

The route re-raises existing `HTTPException` instances so their intended
status code and detail survive. Any other exception becomes a 500 response
whose detail starts with `Unexpected error during prediction:`.

### 3. Filename validation

The service rejects a missing filename with HTTP 400. It then calls
`validate_image()`:

```text
filename.rsplit(".", 1)[-1].lower()
```

The resulting extension must be one of `jpg`, `jpeg`, or `png`. The extension
check is case-insensitive, so names such as `leaf.JPG` are accepted.

### 4. Temporary storage

The service creates `api/uploads/` if necessary. It prefixes the original file
name with a UUID hex string:

```text
<uuid_without_hyphens>_<original_filename>
```

The complete uploaded stream is read from `file.file` and written to that
temporary path.

### 5. Image preprocessing

`preprocess_image()` performs these operations:

1. Opens the temporary file with `PIL.Image.open`.
2. Converts it to RGB, normalizing grayscale or compatible image modes to
   three color channels.
3. Resizes it to `IMAGE_SIZE`, currently `(380, 380)`.
4. Converts it to a NumPy array.
5. Adds a batch dimension, producing an array shaped like
   `(1, 380, 380, 3)`.
6. Applies TensorFlow EfficientNet `preprocess_input`.

Unreadable or corrupted image content produces HTTP 400, even if the file
extension was allowed.

### 6. Model inference

The processed batch is passed to `disease_model.predict()`. The underlying
Keras model is invoked with `verbose=0`, so per-request prediction progress is
not printed.

The serialized model currently has a two-dimensional output with one row and
38 prediction scores, shaped `(1, 38)`. The local `CLASS_NAMES` list currently
contains 37 labels. This is a model/data contract mismatch: if the model emits
class index 37, decoding can fail because there is no corresponding label.
Resolve that mismatch before treating all 38 model outputs as supported.

### 7. Decoding

`decode_prediction()` uses `numpy.argmax()` to select the highest-scoring
class and `numpy.max()` for its confidence. `get_top_predictions()` sorts the
first output row in descending order and selects the first three indexes.

Both helpers index the same ordered `CLASS_NAMES` list. The position of a
label in that list must match the corresponding output neuron in the model.

### 8. Cleanup

After a successful decode, the service checks whether the temporary file
exists and removes it. A cleanup failure is silently ignored by the current
implementation. If an earlier stage raises, the current code does not execute
the later cleanup block, so failed requests can leave files in `uploads/`.

---

## Files explained

### `main.py`

**Responsibility:** Build and expose the FastAPI application.

Important behavior:

- Sets `TF_USE_LEGACY_KERAS` before importing the model and image utilities.
- Defines an `asynccontextmanager` lifespan function.
- Loads the shared disease model before serving requests.
- Sets application metadata:
  - Title: `AGRI Disease Detection API`
  - Description: `Plant disease detection using EfficientNet-B4`
  - Version: `1.0.0`
- Defines `GET /`.
- Defines `GET /health`.
- Includes the router from `routes/disease_routes.py`.

The module-level `app` object is the object passed to Uvicorn:

```powershell
uvicorn api.main:app
```

### `config.py`

**Responsibility:** Centralize static paths and prediction constants.

| Name | Current value | Meaning |
|---|---|---|
| `BASE_DIR` | directory containing `config.py` | Absolute API directory |
| `IMAGE_SIZE` | `(380, 380)` | Width and height expected by preprocessing |
| `MODEL_DIR` | `api/models/pretrained` | Model artifact directory |
| `MODEL_PATH` | `.../plant_disease_efficientnetb4.h5` | Absolute model path |
| `UPLOAD_FOLDER` | `api/uploads` | Temporary upload directory |
| `ALLOWED_EXTENSIONS` | `{"jpg", "jpeg", "png"}` | Accepted filename suffixes |
| `CONFIDENCE_THRESHOLD` | `0.50` | Defined threshold, currently unused |

Paths are built from `BASE_DIR`, not from the current shell directory. This
allows the API to locate its model and upload directory when started from the
repository root with `api.main:app`.

### `routes/disease_routes.py`

**Responsibility:** Declare the disease prediction HTTP route.

The router uses:

- Prefix: `/disease`
- OpenAPI tag: `disease-detection`
- Method and path: `POST /disease/predict`

FastAPI's `File(...)` declaration makes the upload required and causes the
framework to parse multipart form data. The route itself is asynchronous, but
it calls the synchronous service directly.

### `services/disease_service.py`

**Responsibility:** Coordinate the complete prediction use case.

`predict_disease(file)` is the service entry point. It owns:

- filename presence validation;
- extension validation;
- upload directory creation;
- UUID-prefixed temporary file naming;
- file persistence;
- preprocessing;
- model inference;
- result decoding;
- top-three result construction;
- successful-path cleanup.

It translates failures at each major stage into an `HTTPException` with a
stage-specific message:

| Stage | Status | Detail prefix |
|---|---:|---|
| Save upload | 500 | `Failed to save uploaded image:` |
| Preprocess | 500, unless already an HTTP error | `Image preprocessing failed:` |
| Inference | 500 | `Model inference failed:` |
| Decode | 500 | `Prediction decoding failed:` |

### `utils/image_utils.py`

**Responsibility:** Validate names and convert files into model input.

`validate_image(filename)` checks only the final suffix. It does not inspect
the MIME type or file contents.

`preprocess_image(image_path)` is responsible for image decoding and tensor
preparation. It raises a 400 error for unreadable/corrupt content and returns
a NumPy array ready for Keras inference.

The module sets `TF_USE_LEGACY_KERAS=1` before importing TensorFlow's
EfficientNet preprocessing function. This matches the environment setup in
`main.py` and `disease_model.py`.

### `utils/prediction_utils.py`

**Responsibility:** Convert numeric model output into API-friendly objects.

`decode_prediction(predictions)` returns:

```python
{
    "disease": "<label from CLASS_NAMES>",
    "confidence": <float rounded to four places>,
}
```

`get_top_predictions(predictions, top_k=3)` returns a list of:

```python
{
    "label": "<label from CLASS_NAMES>",
    "confidence": <float rounded to four places>,
}
```

These functions assume a batch dimension and read the first prediction row
when building the top list.

### `models/disease_model.py`

**Responsibility:** Load and reuse the TensorFlow model.

`DiseaseModel` implements a simple singleton:

- `_instance` stores the one Python wrapper instance.
- `_model` stores the loaded Keras model.
- `__new__()` ensures repeated construction returns the same wrapper.
- `load()` does nothing if `_model` is already populated.
- `predict()` loads on demand if needed, then calls Keras `predict()`.

The model is loaded with:

```python
tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
    safe_mode=False,
)
```

`compile=False` avoids requiring training/optimizer state for inference.
`safe_mode=False` permits the current serialized model format to load, so the
model file must be treated as trusted application input.

The module exports the shared instance:

```python
disease_model = DiseaseModel()
```

### `data/labels.py`

**Responsibility:** Define the output vocabulary.

`CLASS_NAMES` is currently an ordered list of 37 strings. The list is data, not a
classifier: it contains no prediction logic. Its ordering is part of the
model contract and must not be rearranged without a matching model retraining
or output mapping update.

### `__init__.py` files

The empty `__init__.py` files mark `api`, `data`, `models`, `routes`,
`services`, and `utils` as Python packages. They do not register routes,
configure globals, or run initialization code.

### `models/pretrained/plant_disease_efficientnetb4.h5`

**Responsibility:** Store the serialized EfficientNet-B4 Keras model used for
inference.

The application expects this exact file name in this exact directory. If it is
missing, model startup fails with a message that points to the Hugging Face
PlantDiseaseDetection repository.

### `models/pretrained/.cache/`

This is model-download metadata/cache content. It is not imported directly and
does not define runtime configuration.

### `uploads/`

This directory is created on demand. Uploaded files are given generated names
and are intended to be removed after successful prediction. Do not use it as a
permanent media directory.

---

## HTTP API

### `GET /`

Returns a minimal process-running message:

```json
{
  "message": "AGRI API Running"
}
```

This is a basic smoke-check endpoint. It does not verify that the model is
loaded or that inference is available.

### `GET /health`

Returns:

```json
{
  "status": "healthy"
}
```

This is currently a static health response. It does not perform a model
prediction or inspect the internal model object.

### `POST /disease/predict`

Upload a leaf image as multipart form data:

```text
POST /disease/predict
Content-Type: multipart/form-data
field: file
```

Successful response:

```json
{
  "disease": "Tomato___Late_blight",
  "confidence": 0.9843,
  "top_predictions": [
    {
      "label": "Tomato___Late_blight",
      "confidence": 0.9843
    },
    {
      "label": "Tomato___Early_blight",
      "confidence": 0.012
    },
    {
      "label": "Tomato___Leaf_Mold",
      "confidence": 0.003
    }
  ]
}
```

The endpoint does not currently expose a response schema class, so the exact
shape is produced by the returned Python dictionary.

---

## Prediction response

### Confidence values

The API reports the raw model scores as floating-point values rounded to four
decimal places. It does not independently normalize, calibrate, or threshold
the scores.

For a standard softmax classifier, the scores are normally interpreted as
class probabilities and are expected to sum approximately to one. That
assumption comes from the model; the API does not enforce it.

### Top-three ordering

`top_predictions` is ordered from highest score to lowest score. It contains
up to three entries based on the output array. With the current 38-output model
and 37-label list, predictions whose highest indexes are 0 through 36 can
produce three entries, but any attempt to decode index 37 can fail.

### Label formatting

Labels retain the PlantVillage-style names, including underscores, commas,
parentheses, and spaces. The API does not convert them into display-friendly
names.

---

## Validation and error behavior

Typical errors include:

| Situation | Expected status | Example detail |
|---|---:|---|
| No multipart `file` field | 422 | FastAPI validation response |
| Empty/missing filename | 400 | `No filename provided.` |
| Unsupported extension | 400 | `Unsupported file type 'gif'. Allowed: jpeg, jpg, png` |
| Corrupt or unreadable image | 400 | `Corrupted or unreadable image: ...` |
| Cannot write upload | 500 | `Failed to save uploaded image: ...` |
| Preprocessing failure | 500 | `Image preprocessing failed: ...` |
| Model inference failure | 500 | `Model inference failed: ...` |
| Result decoding failure | 500 | `Prediction decoding failed: ...` |
| Other route/service failure | 500 | `Unexpected error during prediction: ...` |

If the model file is absent, the startup lifespan raises `FileNotFoundError`.
The server should be considered not ready until model loading completes.

---

## Model and label mapping

The bundled model is an EfficientNet-B4 classifier whose serialized final
layer has 38 softmax outputs. The local label list currently contains 37
entries, so the implementation needs a label/model alignment fix before every
model output can be decoded safely.
The expected class order is:

1. `Apple___Apple_scab`
2. `Apple___Black_rot`
3. `Apple___Cedar_apple_rust`
4. `Apple___healthy`
5. `Blueberry___healthy`
6. `Cherry___Powdery_mildew`
7. `Cherry___healthy`
8. `Corn___Cercospora_leaf_spot Gray_leaf_spot`
9. `Corn___Common_rust`
10. `Corn___Northern_Leaf_Blight`
11. `Corn___healthy`
12. `Grape___Black_rot`
13. `Grape___Esca_(Black_Measles)`
14. `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)`
15. `Grape___healthy`
16. `Orange___Haunglongbing_(Citrus_greening)`
17. `Peach___Bacterial_spot`
18. `Pepper,_bell___Bacterial_spot`
19. `Pepper,_bell___healthy`
20. `Potato___Early_blight`
21. `Potato___Late_blight`
22. `Potato___healthy`
23. `Raspberry___healthy`
24. `Soybean___healthy`
25. `Squash___Powdery_mildew`
26. `Strawberry___Leaf_scorch`
27. `Strawberry___healthy`
28. `Tomato___Bacterial_spot`
29. `Tomato___Early_blight`
30. `Tomato___Late_blight`
31. `Tomato___Leaf_Mold`
32. `Tomato___Septoria_leaf_spot`
33. `Tomato___Spider_mites Two-spotted_spider_mite`
34. `Tomato___Target_Spot`
35. `Tomato___Tomato_Yellow_Leaf_Curl_Virus`
36. `Tomato___Tomato_mosaic_virus`
37. `Tomato___healthy`

The source list currently contains 37 entries; the numbered list above follows
its current order and should be checked against `data/labels.py` whenever
labels are changed. In particular, never alphabetize the list as a cosmetic change:
the numeric index-to-label relationship is what makes predictions meaningful.

> **Maintenance check:** The current model/list count mismatch must be resolved
> first. If a replacement model has a different class order, update
> `CLASS_NAMES` to the model's exact output order and validate the mapping with
> known sample images before deploying it.

---

## Configuration reference

Configuration is code-based in `config.py`; there are currently no API
environment variables for these values.

### Changing the image size

Update `IMAGE_SIZE` only when the replacement model expects a different input
resolution. The preprocessing output and the model input shape must agree.

### Changing the model file

Replace the artifact at `MODEL_PATH` or update `MODEL_PATH` to another local
file. Confirm that:

- the file can be loaded by the installed TensorFlow/Keras versions;
- the model accepts RGB batches;
- the expected input size matches `IMAGE_SIZE`;
- the output class count matches `CLASS_NAMES`;
- the output order matches `CLASS_NAMES`.

### Changing accepted file types

Update `ALLOWED_EXTENSIONS` and test both extension validation and actual image
decoding. Adding an extension does not make Pillow capable of decoding every
file with that suffix.

### Confidence threshold

`CONFIDENCE_THRESHOLD` is defined for possible future policy use but is not
referenced by the current service or prediction helpers. Changing its value
does not currently change API responses.

---

## Local setup and startup

Run commands from the repository root, not from inside `api/`, because the
application is imported as the package `api`.

### 1. Use the project environment

PowerShell:

```powershell
.venv\Scripts\activate
```

The project declares Python `>=3.13` and its dependencies in the root
`pyproject.toml`. The API requires, at minimum, FastAPI, Uvicorn,
`python-multipart`, Pillow, NumPy, TensorFlow, and the legacy Keras
compatibility package already declared there.

### 2. Start the development server

```powershell
$env:TF_USE_LEGACY_KERAS="1"
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

The environment variable is also set by the relevant Python modules before
their TensorFlow imports, but setting it in the shell makes the startup intent
explicit.

Wait for a log line equivalent to:

```text
Model loaded successfully.
```

The `--reload` option is for local development. It can create a reloader
process and a worker process, so model loading messages may appear more than
once.

### 3. Open generated documentation

FastAPI automatically exposes:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

The disease router's `disease-detection` tag appears in these documents.

---

## Testing the running API

### Health smoke checks

```powershell
Invoke-RestMethod http://127.0.0.1:8000/
Invoke-RestMethod http://127.0.0.1:8000/health
```

Expected results:

```text
message          status
-------          ------
AGRI API Running
```

and:

```text
status
------
healthy
```

These checks verify routing only; they do not prove that a real image can be
classified.

### Prediction with cURL

From the repository root:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/disease/predict" `
  -F "file=@path\to\leaf_image.jpg"
```

### Prediction with PowerShell

For a simple multipart request:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/disease/predict" `
  -F "file=@api\uploads\sample.jpg"
```

Use a test image outside `api/uploads/` when possible, because that directory
is intended for service-managed temporary files.

### Negative-path checks

Verify at least these cases when changing validation or error handling:

```text
1. Omit the file field                       -> 422
2. Upload a .gif or .txt file                -> 400
3. Rename a non-image file to .jpg           -> 400
4. Upload a valid JPG or PNG                 -> 200
5. Temporarily move the model artifact       -> startup failure
```

Do not use a file extension alone as proof that the file is a valid image; the
corrupt-image case exercises Pillow decoding.

---

## Operational notes

### CPU and GPU detection

At model load time, `disease_model.py` prints the TensorFlow version and the
number/list of physical GPUs detected by TensorFlow. The code does not select a
GPU explicitly or configure GPU memory growth.

### Memory usage

The model is held in process memory after loading. Each request also creates a
temporary decoded image and a NumPy batch. Multiple Uvicorn worker processes
will each load their own model copy.

### Concurrency

The route and service call synchronous file I/O, Pillow processing, and
TensorFlow inference directly from an `async def` route. This is adequate for
the current simple service, but expensive inference can block the event loop
under concurrent load.

### Temporary files

Successful requests remove their temporary files. Failed requests can leave
files behind because cleanup is not in a `finally` block. Monitor disk usage
and periodically remove stale files if the API is exposed to repeated invalid
or failing requests.

### Logging

Model startup currently uses `print()` for TensorFlow, GPU, and model-loading
messages. Request failures are returned through HTTP exceptions. There is no
structured request ID, access log customization, metrics endpoint, or
prediction audit log in this folder.

### Security boundary

The API has no authentication or authorization. Anyone who can reach the
server can call the prediction endpoint. Put it behind an appropriate trusted
network boundary or add authentication before exposing it publicly.

---

## Troubleshooting

### `Model file not found`

Confirm that this path exists relative to the repository:

```text
api/models/pretrained/plant_disease_efficientnetb4.h5
```

The path is resolved from `config.py`, so changing the shell's current
directory should not change the expected model location.

### TensorFlow/Keras import or loading errors

Check that the active interpreter is the project environment and that the
versions in the root `pyproject.toml` are installed. Keep
`TF_USE_LEGACY_KERAS=1` set before starting the server. A model serialized
with incompatible TensorFlow/Keras versions may need to be re-exported or run
with its original compatible environment.

### `python-multipart` or form parsing errors

FastAPI needs `python-multipart` to parse file uploads. Install the project
dependencies from the root manifest if the package is missing.

### Valid-looking file rejected as corrupt

The extension validator only checks the suffix; Pillow performs the actual
decode. Confirm that the file is a real JPG/PNG, is not truncated, and can be
opened by an image viewer. A renamed text or unsupported-format file will fail.

### Prediction labels look incorrect

Verify that the replacement model's output neuron order exactly matches
`data/labels.py`. A model can load successfully while producing semantically
wrong labels if the list order is mismatched.

### `uploads/` contains old files

This can happen when save, preprocessing, inference, or decoding fails before
the current cleanup block runs. Remove only known stale files after confirming
they are not being used by an active request.

### `/health` is healthy but prediction fails

This is possible by design: `/health` returns a static payload and does not
exercise model inference. Use a real multipart prediction request and inspect
the server startup log for model-loading errors.

---

## Extension and maintenance guide

### Add another endpoint

1. Add a focused route module under `routes/`.
2. Put orchestration/business behavior in `services/`.
3. Reuse or add narrowly scoped helpers under `utils/`.
4. Register the router in `main.py`.
5. Verify the route appears in `/docs`.
6. Document the endpoint and its errors in this README.

### Add a new disease class

Adding a class requires more than appending a string:

1. Train or obtain a model with the new output class.
2. Confirm the exact output order.
3. Update `CLASS_NAMES` to the exact order.
4. Confirm preprocessing and input shape.
5. Test representative images for old and new classes.
6. Update the class-count and model documentation here.

### Replace the model

Treat the model file, preprocessing, and labels as one versioned contract.
Validate them together. A replacement model that has the same file extension
and loads without error is not necessarily compatible with this service.

### Improve cleanup reliability

If modifying the service, place temporary-file removal in a `finally` block
around the processing stages. Preserve the original exception/status while
ensuring cleanup runs after both success and failure.

### Add confidence handling

If `CONFIDENCE_THRESHOLD` is activated, document the response contract first:
for example, whether low-confidence predictions should return a new status,
an explicit `uncertain` field, or still return the best class. The current API
always returns the highest-scoring class regardless of score.

---

## Current limitations

The current implementation is intentionally small, but the following behavior
is important when deploying or extending it:

- Validation is based on filename extension before content decoding.
- Uploads are read fully into memory and written synchronously.
- The route invokes synchronous inference from an async handler.
- Temporary-file cleanup is performed only after successful decoding.
- Cleanup errors are ignored.
- `/health` is static and does not represent model readiness.
- `CONFIDENCE_THRESHOLD` is declared but unused.
- There is no authentication, rate limiting, upload-size limit, or content
  scanning in this folder.
- There is no persistent storage of requests or predictions.
- Error details include underlying exception text.
- There is no explicit response model enforcing the JSON schema.
- Each worker process loads its own model instance.
- Labels are returned in raw PlantVillage naming format.
- The serialized model exposes 38 outputs while `CLASS_NAMES` exposes 37
  labels; this must be reconciled before deployment.

These are descriptions of the code as it exists, not claims that the API is
production-hardened. Address the relevant items before exposing the service to
untrusted public traffic.

---

## Quick reference

| Item | Value |
|---|---|
| App import | `api.main:app` |
| Start command | `uvicorn api.main:app --reload --host 127.0.0.1 --port 8000` |
| Root endpoint | `GET /` |
| Health endpoint | `GET /health` |
| Prediction endpoint | `POST /disease/predict` |
| Upload field | `file` |
| Accepted suffixes | `.jpg`, `.jpeg`, `.png` |
| Model input | RGB batch shaped `(1, 380, 380, 3)` |
| Model artifact | `models/pretrained/plant_disease_efficientnetb4.h5` |
| Serialized model outputs | 38 |
| Local label entries | 37 (currently mismatched) |
| Top results | 3 |
| Temporary directory | `uploads/` |
| API docs | `/docs`, `/redoc`, `/openapi.json` |
