import tensorflow as tf
import base64
from io import BytesIO
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import gdown

# ==============================
# CONFIGURATION
# ==============================
IMG_SIZE = 224
MODEL_PATH = "EfficientNetB4_72classes_v2.tflite"
TREATMENT_PATH = "plant_recommendations.xlsx"

# ── Replace with your actual Google Drive file IDs ──
MODEL_DRIVE_ID = "11sJCR3K2c-l3jayK1OE4rKoB0vUyClQG"
TREATMENT_DRIVE_ID  = "1SM6OkNffM4Z1vznd5PolWL4i9K2HHonG"

CLASS_NAMES = [
    "Apple___alternaria_leaf_spot", "Apple___black_rot", "Apple___brown_spot",
    "Apple___gray_spot", "Apple___healthy", "Apple___rust", "Apple___scab",
    "Bell_pepper___bacterial_spot", "Bell_pepper___healthy", "Blueberry___healthy",
    "Cassava___bacterial_blight", "Cassava___brown_streak_disease", "Cassava___green_mottle",
    "Cassava___healthy", "Cassava___mosaic_disease", "Cherry___healthy",
    "Cherry___powdery_mildew", "Coffee___healthy", "Coffee___red_spider_mite",
    "Coffee___rust", "Corn___common_rust", "Corn___gray_leaf_spot", "Corn___healthy",
    "Corn___northern_leaf_blight", "Grape___Leaf_blight", "Grape___black_measles",
    "Grape___black_rot", "Grape___healthy", "Grape___leaf_blight",
    "Orange___citrus_greening", "Peach___bacterial_spot", "Peach___healthy",
    "Potato___bacterial_wilt", "Potato___early_blight", "Potato___healthy",
    "Potato___late_blight", "Potato___leafroll_virus", "Potato___mosaic_virus",
    "Potato___nematode", "Potato___pests", "Potato___phytophthora",
    "Raspberry___healthy", "Rice___bacterial_blight", "Rice___blast",
    "Rice___brown_spot", "Rice___tungro", "Rose___healthy", "Rose___rust",
    "Rose___slug_sawfly", "Soybean___healthy", "Squash___powdery_mildew",
    "Strawberry___healthy", "Strawberry___leaf_scorch", "Sugercane___healthy",
    "Sugercane___mosaic", "Sugercane___red_rot", "Sugercane___rust",
    "Sugercane___yellow_leaf", "Tomato___bacterial_spot", "Tomato___early_blight",
    "Tomato___healthy", "Tomato___late_blight", "Tomato___leaf_curl",
    "Tomato___leaf_mold", "Tomato___mosaic_virus", "Tomato___septoria_leaf_spot",
    "Tomato___spider_mites", "Tomato___target_spot", "Watermelon___anthracnose",
    "Watermelon___downy_mildew", "Watermelon___healthy", "Watermelon___mosaic_virus"
]

app = FastAPI(title="Smart Plant Disease Diagnostics API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

interpreter = None
treatment_df = None

# ==============================
# HELPER: Google Drive Download
# ==============================
def download_from_drive(file_id: str, output_path: str, description: str, is_sheet: bool = False):
    """Downloads a file from Google Drive if it doesn't already exist locally."""
    if not os.path.exists(output_path):
        print(f"Downloading {description} from Google Drive...")
        if is_sheet:
            url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=xlsx"
            gdown.download(url, output_path, quiet=False, fuzzy=True)
        else:
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output_path, quiet=False)
        print(f"{description} downloaded successfully.")
    else:
        print(f"{description} already exists locally, skipping download.")
# ==============================
# STARTUP
# ==============================
@app.on_event("startup")
async def load_resources():
    global interpreter, treatment_df

    # ── Step 1: Download files from Drive if not present ──
    try:
        download_from_drive(MODEL_DRIVE_ID,     MODEL_PATH,     "TFLite Model",       is_sheet=False)
        download_from_drive(TREATMENT_DRIVE_ID, TREATMENT_PATH, "Treatment Database", is_sheet=True)
    except Exception as e:
        print(f"Error downloading files from Google Drive: {e}")

    # ── Step 2: Load TFLite model ──
    try:
        interpreter = tf.lite.Interpreter(
            model_path=MODEL_PATH,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.AUTO
        )
        interpreter.allocate_tensors()
        print("TFLite model loaded successfully.")
    except Exception as e:
        print(f"Error loading TFLite model: {e}")

    # ── Step 3: Load treatment database ──
    try:
        df = pd.read_excel(TREATMENT_PATH)
        df.columns = df.columns.str.lower().str.strip()
        if "disease" in df.columns:
            df["disease"] = (
                df["disease"]
                .astype(str)
                .str.lower()
                .str.replace("___", " ", regex=False)
                .str.replace("_", " ", regex=False)
            )
            df["disease"] = df["disease"].apply(lambda x: ' '.join(x.split()))
        treatment_df = df
        print("Treatment database loaded successfully.")
    except Exception as e:
        print(f"Error loading treatment database: {e}")
        treatment_df = pd.DataFrame(columns=["disease"])

# ==============================
# HEALTH CHECK ENDPOINT
# ==============================
@app.get("/")
def home():
    return {
        "message": "API is running",
        "model_loaded": interpreter is not None,
        "database_loaded": treatment_df is not None and not treatment_df.empty
    }

# ==============================
# CORE FUNCTIONS
# ==============================
def preprocess_image(image: Image.Image):
    img = image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_disease(img_array):
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img_array = img_array.astype(input_details[0]['dtype'])

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    top5_indices = np.argsort(predictions)[::-1][:5]
    print("\nTop 5 Predictions:")
    for i in top5_indices:
        print(f"  {CLASS_NAMES[i]}: {predictions[i]*100:.2f}%")

    predicted_index = top5_indices[0]
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(100 * predictions[predicted_index])

    return predicted_class, confidence, predictions


def calculate_severity_by_color(img: Image.Image, disease_name: str):
    if "healthy" in disease_name.lower():
        return 0.0, "Healthy", None

    img_cv = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2HSV)

    lower_green     = np.array([25, 30, 30])
    upper_green     = np.array([90, 255, 255])
    leaf_mask       = cv2.inRange(hsv, lower_green, upper_green)
    dark_green_mask = cv2.inRange(hsv, np.array([25, 10, 10]), np.array([90, 100, 100]))
    leaf_mask       = cv2.bitwise_or(leaf_mask, dark_green_mask)

    leaf_pixels = cv2.countNonZero(leaf_mask)
    if leaf_pixels < 100:
        leaf_mask   = np.ones(img_cv.shape[:2], dtype=np.uint8) * 255
        leaf_pixels = IMG_SIZE * IMG_SIZE

    brown_mask  = cv2.inRange(hsv, np.array([5,  40,  40]),  np.array([25, 255, 200]))
    yellow_mask = cv2.inRange(hsv, np.array([20, 40,  100]), np.array([35, 255, 255]))
    white_mask  = cv2.inRange(hsv, np.array([0,  0,   180]), np.array([180, 40, 255]))
    dark_mask   = cv2.inRange(hsv, np.array([0,  0,   0]),   np.array([180, 255, 60]))

    disease_mask = cv2.bitwise_or(brown_mask, yellow_mask)
    disease_mask = cv2.bitwise_or(disease_mask, white_mask)
    disease_mask = cv2.bitwise_or(disease_mask, dark_mask)

    actual_infected = cv2.bitwise_and(disease_mask, leaf_mask)
    kernel          = np.ones((3, 3), np.uint8)
    actual_infected = cv2.morphologyEx(actual_infected, cv2.MORPH_OPEN, kernel)

    infected_pixels = cv2.countNonZero(actual_infected)
    infection_ratio = float(min((infected_pixels / leaf_pixels) * 100, 100.0))

    if infection_ratio <= 15:
        severity_level = "Mild"
    elif infection_ratio <= 40:
        severity_level = "Moderate"
    else:
        severity_level = "Severe"

    overlay  = img_cv.copy()
    overlay[actual_infected > 0] = [255, 0, 0]
    blended  = cv2.addWeighted(img_cv, 0.6, overlay, 0.4, 0)
    contours, _ = cv2.findContours(actual_infected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (255, 165, 0), 2)

    return infection_ratio, severity_level, blended


def encode_image_to_base64(img_array):
    img_pil  = Image.fromarray(np.uint8(img_array))
    buffered = BytesIO()
    img_pil.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def smart_get_treatment(disease_name):
    clean_pred = disease_name.lower().replace("___", " ").replace("_", " ")
    clean_pred = ' '.join(clean_pred.split())
    translation_dict = {
        "apple alternaria leaf spot": "apple alternaria leaf sp",
        "watermelon mosaic virus":    "watermelon mosaic vir",
        "grape leaf blight":          "grape leaf blight"
    }
    if clean_pred in translation_dict:
        clean_pred = translation_dict[clean_pred]
    exact_match = treatment_df[treatment_df["disease"] == clean_pred]
    if not exact_match.empty:
        return exact_match.iloc[0].fillna("Not available").to_dict()
    for _, row in treatment_df.iterrows():
        db_disease = str(row["disease"])
        if clean_pred in db_disease or db_disease in clean_pred:
            return row.fillna("Not available").to_dict()
    return None

# ==============================
# API ENDPOINT
# ==============================
@app.post("/diagnose")
async def diagnose_plant(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    # Guard: ensure model and database are loaded
    if interpreter is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please try again shortly.")
    if treatment_df is None or treatment_df.empty:
        raise HTTPException(status_code=503, detail="Treatment database not loaded yet.")

    try:
        contents = await file.read()
        img      = Image.open(BytesIO(contents)).convert("RGB")
        img_array = preprocess_image(img)

        disease, confidence, _          = predict_disease(img_array)
        clean_name                      = disease.replace('___', ' - ').replace('_', ' ').title()
        infection_ratio, severity_level, overlay_img = calculate_severity_by_color(img, disease)
        cam_base64                      = encode_image_to_base64(overlay_img) if overlay_img is not None else None
        treatment_info                  = smart_get_treatment(disease)

        return {
            "status": "success",
            "diagnosis": {
                "raw_class":          disease,
                "clean_name":         clean_name,
                "confidence_percent": round(confidence, 2),
                "is_healthy":         "healthy" in disease.lower()
            },
            "severity_metrics": {
                "infection_ratio_percent": round(infection_ratio, 2),
                "severity_level":          severity_level
            },
            "treatment_protocol":  treatment_info,
            "xai_overlay_base64":  cam_base64
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
