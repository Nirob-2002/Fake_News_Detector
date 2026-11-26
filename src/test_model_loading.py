
import json
import re
import unicodedata
import tensorflow as tf
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models" / "deep_learning"
RESULT_DIR = BASE_DIR / "results" / "deep_learning"

# Same preprocessing used during training
def clean_text(text, language):

    text = str(text)
    text = unicodedata.normalize("NFC", text)

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    text = re.sub(r"<[^>]+>", " ", text)

    text = re.sub(
        r"[\u200b\ufeff]",
        "",
        text
    )

    if language == "English":
        text = text.lower()

    text = re.sub(
        r"[^\w\s\u0980-\u09FF]",
        " ",
        text,
        flags=re.UNICODE
    )

    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


print("=" * 50)
print("FAKE NEWS MODEL LOADING TEST")
print("=" * 50)

for language in ["english", "bangla"]:

    print(f"\nLoading {language.upper()} BiLSTM...")

    model_path = MODEL_DIR / f"{language}_bilstm.keras"

    threshold_path = (
        RESULT_DIR /
        f"{language}_bilstm_threshold.json"
    )

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    with open(threshold_path, "r", encoding="utf-8") as f:
        threshold_data = json.load(f)

    threshold = float(threshold_data["threshold"])

    print("Model loaded successfully!")
    print("Threshold:", threshold)

    if language == "english":

        sample = (
            "The government announced a new economic "
            "development program for the country."
        )

    else:

        sample = (
            "বাংলাদেশ সরকার দেশের অর্থনৈতিক উন্নয়নের "
            "জন্য নতুন কর্মসূচি ঘোষণা করেছে।"
        )

    processed = clean_text(
        sample,
        language.capitalize()
    )

    input_tensor = tf.constant(
    [processed],
    dtype=tf.string
    )

    prediction_output = model(
    input_tensor,
    training=False
)

    probability = float(
    prediction_output.numpy().reshape(-1)[0]
    )

    prediction = (
        "Fake" if probability >= threshold else "Real"
    )

    print("Prediction:", prediction)
    print("Fake probability:", round(probability * 100, 2), "%")

    del model

print("\nALL MODEL TESTS COMPLETED!")
