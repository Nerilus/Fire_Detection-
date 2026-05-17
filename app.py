import io
import base64
import numpy as np
import matplotlib
import matplotlib.cm as cm
from flask import Flask, request, jsonify, render_template
from PIL import Image
import tensorflow as tf

matplotlib.use("Agg")  # moteur non-interactif (pas de fenetre GUI)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 Mo max

MODEL_PATH = "fire_model.keras"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp", "gif"}

# Mapping des classes (flow_from_directory, ordre alphabetique)
# fire_images -> 0  |  non_fire_images -> 1
# sigmoid output ~ P(non_fire) : proche de 0 = feu, proche de 1 = pas de feu

print("Chargement du modele...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Modele charge avec succes.")

# Pre-construire le feature model pour Grad-CAM (une seule fois)
try:
    _mobilenet = model.get_layer("mobilenetv2_1.00_224")
    _feature_model = tf.keras.Model(
        inputs=_mobilenet.inputs,
        outputs=_mobilenet.get_layer("out_relu").output,
    )
    GRADCAM_AVAILABLE = True
    print("Grad-CAM disponible.")
except Exception as e:
    GRADCAM_AVAILABLE = False
    print(f"Grad-CAM non disponible : {e}")


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def preprocess_image(image_bytes):
    """Charge, redimensionne et normalise une image pour le modele (224x224, [0,1])."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    return np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)


def compute_gradcam(img_array, is_fire):
    """
    Calcule la heatmap Grad-CAM pour la classe predite.

    Strategie :
      1. Extraire les activations de out_relu  (7x7x1280)  hors de la tape
      2. Regarder ce tenseur avec la tape
      3. Appliquer manuellement GAP + couches denses pour reconstruire le graphe
      4. Calculer le gradient  dLoss/dConv
      5. Ponderer les canaux et appliquer ReLU

    Retourne : numpy array (7, 7) normalise dans [0, 1], ou None si erreur.
    """
    if not GRADCAM_AVAILABLE:
        return None
    try:
        img_tensor = tf.cast(img_array, tf.float32)

        # Etape 1 : activations de la derniere couche conv (hors tape)
        conv_features = _feature_model(img_tensor)  # (1, 7, 7, 1280)

        # Etape 2 : rejouer le forward pass depuis conv_features avec la tape
        with tf.GradientTape() as tape:
            tape.watch(conv_features)

            # GlobalAveragePooling2D (derniere operation dans MobileNetV2)
            x = tf.reduce_mean(conv_features, axis=[1, 2])  # (1, 1280)

            # Couches denses de l'exterieur : Dense(256) -> Dropout -> Dense(1)
            for layer in model.layers[2:]:
                x = layer(x, training=False)

            predictions = x  # (1, 1)
            loss = (1.0 - predictions[:, 0]) if is_fire else predictions[:, 0]

        grads = tape.gradient(loss, conv_features)  # (1, 7, 7, 1280)
        if grads is None:
            return None

        # Etape 3 : pooler les gradients et ponderer les cartes
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))      # (1280,)
        heatmap = tf.reduce_sum(conv_features[0] * pooled_grads, axis=-1)  # (7, 7)
        heatmap = tf.nn.relu(heatmap)

        max_val = tf.reduce_max(heatmap)
        heatmap = heatmap / max_val if max_val > 0 else heatmap

        return heatmap.numpy()

    except Exception as e:
        print(f"Erreur Grad-CAM : {e}")
        return None


def make_overlay(original_bytes, heatmap_np, alpha=0.45):
    """
    Superpose la heatmap colorisee (colormap 'inferno') sur l'image originale.
    Retourne une chaine base64 JPEG.
    """
    # Redimensionner la heatmap (7x7 -> 224x224)
    hm_img = Image.fromarray(np.uint8(255 * heatmap_np))
    hm_img = hm_img.resize((224, 224), Image.LANCZOS)

    # Appliquer la colormap
    colormap = matplotlib.colormaps["inferno"]
    colored = colormap(np.array(hm_img, dtype=np.float32) / 255.0)[:, :, :3]
    hm_colored = Image.fromarray(np.uint8(255 * colored))

    # Charger et redimensionner l'originale
    orig = Image.open(io.BytesIO(original_bytes)).convert("RGB").resize((224, 224))

    # Fusionner
    overlay = Image.blend(orig, hm_colored, alpha=alpha)

    buf = io.BytesIO()
    overlay.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier fourni."}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "Aucun fichier selectionne."}), 400

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Format non supporte : .{ext}"}), 400

    image_bytes = file.read()

    try:
        img_array = preprocess_image(image_bytes)
    except Exception as e:
        return jsonify({"error": f"Impossible de lire l'image : {str(e)}"}), 400

    # --- Inference ---
    raw_score = float(model.predict(img_array, verbose=0)[0][0])
    is_fire = raw_score < 0.5
    confidence = round((1.0 - raw_score if is_fire else raw_score) * 100, 1)

    # --- Grad-CAM ---
    gradcam_b64 = None
    heatmap = compute_gradcam(img_array, is_fire)
    if heatmap is not None:
        gradcam_b64 = "data:image/jpeg;base64," + make_overlay(image_bytes, heatmap)

    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"

    return jsonify({
        "is_fire": is_fire,
        "label": "Feu detecte" if is_fire else "Aucun feu",
        "confidence": confidence,
        "raw_score": round(raw_score * 100, 1),
        "image_data": f"data:{mime};base64," + base64.b64encode(image_bytes).decode("utf-8"),
        "gradcam_data": gradcam_b64,
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
