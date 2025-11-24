# ---------------------------
# File: app/model.py
# ---------------------------
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D

# ---------- GPU setup ----------
# Try to enable memory growth for GPUs to avoid TF pre-allocating all memory.
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception as e:
        # If setting memory growth fails, just print a warning and continue.
        print("Warning: Could not set memory growth:", e)

print("Num GPUs Available:", len(gpus))
print("TensorFlow version:", tf.__version__)

# -------- Load model --------
MODEL_PATH = os.getenv("MODEL_PATH", "saved_model/InceptionV3_Brain_Tumor_MRI.h5")
print("Loading model from:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
model.trainable = False

# -------- Find last conv layer and build grad_model once --------
# Find the last Conv2D layer
last_conv_layer = None
for layer in reversed(model.layers):
    if isinstance(layer, Conv2D):
        last_conv_layer = layer
        break
if last_conv_layer is None:
    raise RuntimeError("No Conv2D layer found in the model; cannot build Grad-CAM.")

target_layer = model.get_layer(last_conv_layer.name)
grad_model = Model(inputs=model.inputs, outputs=[target_layer.output, model.output])
print("Built grad_model with target layer:", target_layer.name)

# -------- Labels --------
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# -------- Preprocessing (use 299x299 for InceptionV3) --------
def preprocess_image_pil(img: Image.Image, target_size=(512, 512)):
    """
    Accepts PIL.Image, returns float32 numpy array shaped (1,H,W,3) with values in [0,1].
    """
    img = img.convert("RGB")
    img = img.resize(target_size, resample=Image.BILINEAR)
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def pil_to_tf_tensor(img: Image.Image, target_size=(512, 512)):
    """
    Convert PIL image to a TF tensor float32 (1,H,W,3) scaled to [0,1].
    Uses TF ops to allow better GPU pipeline.
    """
    arr = preprocess_image_pil(img, target_size=target_size)
    return tf.convert_to_tensor(arr, dtype=tf.float32)

# -------- Prediction helper --------
def predict(img: Image.Image):
    """
    Returns (label, confidence, prob_dict)
    """
    input_tensor = preprocess_image_pil(img)  # numpy (1,H,W,3)
    # Try to call model by direct positional input (works for most Keras models).
    preds = model(input_tensor, training=False)
    probs = preds.numpy()[0]
    class_idx = int(np.argmax(probs))
    confidence = float(np.max(probs))
    prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    return CLASS_NAMES[class_idx], confidence, prob_dict

# --------- Compiled Grad-CAM compute function ---------
# We create a tf.function that computes conv features and gradients for a given input and class index.
@tf.function
def _compute_conv_and_grads(img_input, class_index):
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_input)

        # preds is probably a list -> convert it
        if isinstance(preds, (list, tuple)):
            preds = preds[0]  # take the actual tensor

        class_logits = preds[:, class_index]

    grads = tape.gradient(class_logits, conv_outputs)
    return conv_outputs, grads, preds

def compute_gradcam_overlay(img: Image.Image, interpolant=0.5, target_size=(512,512)):
    """
    High-level wrapper:
    - builds input tensor
    - obtains predicted class index (fast forward)
    - calls compiled grad function to get conv features + grads
    - computes heatmap and overlay efficiently
    Returns: overlay as uint8 HxWx3 numpy array
    """
    # Build tensor
    input_tf = pil_to_tf_tensor(img, target_size=target_size)  # (1,H,W,3), float32

    # Fast predict to get class index (cheap forward pass)
    preds = model(input_tf, training=False)
    pred_np = preds.numpy()[0]
    class_idx = int(np.argmax(pred_np))

    # Use compiled function to compute conv features and grads for that class
    conv_out, grads, _ = _compute_conv_and_grads(input_tf, tf.constant(class_idx, dtype=tf.int64))

    # Convert to numpy and handle shapes robustly
    conv_out_np = conv_out.numpy()
    grads_np = grads.numpy() if grads is not None else None

    if grads_np is None:
        # Fallback: gradients None -> return original image as overlay (no heatmap)
        H = input_tf.shape[1]
        W = input_tf.shape[2]
        original_img = np.array(img.resize((W, H))).astype("uint8")
        if original_img.ndim == 2:
            original_img = np.stack([original_img]*3, axis=-1)
        return original_img

    # conv_out_np shape (1,Hf,Wf,C) -> take first batch
    if conv_out_np.ndim == 4 and conv_out_np.shape[0] == 1:
        conv_out_np = conv_out_np[0]
    # grads_np shape (1,Hf,Wf,C)
    if grads_np.ndim == 4 and grads_np.shape[0] == 1:
        grads_np = grads_np[0]

    # Global average pooling of gradients over spatial dims (Hf,Wf)
    pooled_grads = np.mean(grads_np, axis=(0,1))  # shape (C,)

    # Weighted sum of conv feature maps
    heatmap = np.sum(conv_out_np * pooled_grads[np.newaxis, np.newaxis, :], axis=-1)  # (Hf,Wf)
    heatmap = np.maximum(heatmap, 0.0)
    max_val = np.max(heatmap) if heatmap.size else 0.0
    if max_val > 0:
        heatmap = heatmap / (max_val + 1e-9)
    else:
        heatmap = np.zeros_like(heatmap, dtype=np.float32)

    # Resize heatmap to original image size
    H = input_tf.shape[1]
    W = input_tf.shape[2]
    original_img = np.array(img.resize((W, H))).astype("float32")
    if original_img.ndim == 2:
        original_img = np.stack([original_img]*3, axis=-1)

    heatmap_resized = cv2.resize((heatmap * 255.0).astype("uint8"), (W, H))
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)  # BGR
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB).astype("float32")

    # Ensure original image is in uint8 0..255
    orig_uint8 = np.clip(original_img, 0, 255).astype("uint8")

    # Combine using interpolant: (interpolant * original + (1-interpolant) * heatmap_color)
    overlay = np.clip(orig_uint8.astype("float32") * interpolant + heatmap_color * (1.0 - interpolant), 0, 255).astype("uint8")
    return overlay

# Expose functions for main.py
__all__ = ["model", "grad_model", "predict", "compute_gradcam_overlay", "CLASS_NAMES"]
# Backwards-compatible function name expected by main.py
def gradcam(img: Image.Image, interpolant=0.5):
    return compute_gradcam_overlay(img, interpolant=interpolant)

# ---------------------------
# End of app/model.py
# ---------------------------