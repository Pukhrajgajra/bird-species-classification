import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

def get_gradcam_heatmap(model, img_array, last_conv_layer_name="top_conv"):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), predictions.numpy()

def overlay_gradcam(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    superimposed = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
    return img, heatmap_colored, superimposed

def visualize_gradcam(model, img_path, class_names, save_path=None):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array / 255.0, axis=0)

    heatmap, predictions = get_gradcam_heatmap(model, img_array)
    original, heatmap_img, superimposed = overlay_gradcam(img_path, heatmap)

    top5_idx = np.argsort(predictions[0])[-5:][::-1]
    top5_classes = [(class_names[i], float(predictions[0][i])) for i in top5_idx]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(heatmap_img)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=14)
    axes[1].axis("off")

    axes[2].imshow(superimposed)
    title = f"Prediction: {top5_classes[0][0].split('.')[-1]}\n"
    title += f"Confidence: {top5_classes[0][1]:.2%}"
    axes[2].set_title(title, fontsize=12)
    axes[2].axis("off")

    plt.suptitle("Grad-CAM Explainability", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()

    print("\nTop-5 Predictions:")
    for i, (cls, conf) in enumerate(top5_classes):
        print(f"  {i+1}. {cls.split('.')[-1]:40s} {conf:.2%}")

    return top5_classes

def batch_gradcam(model, test_gen, class_names, n_samples=10, save_dir="results/plots"):
    os.makedirs(save_dir, exist_ok=True)
    batch_images, batch_labels = next(iter(test_gen))

    for i in range(min(n_samples, len(batch_images))):
        img_array = np.expand_dims(batch_images[i], axis=0)
        heatmap, predictions = get_gradcam_heatmap(model, img_array)

        pred_idx = np.argmax(predictions[0])
        true_idx = np.argmax(batch_labels[i])
        pred_class = class_names[pred_idx].split(".")[-1]
        true_class = class_names[true_idx].split(".")[-1]
        correct = "CORRECT" if pred_idx == true_idx else "WRONG"

        img = np.uint8(batch_images[i] * 255)
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_colored = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        superimposed = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img)
        axes[0].set_title(f"True: {true_class}", fontsize=10)
        axes[0].axis("off")
        axes[1].imshow(heatmap_colored)
        axes[1].set_title("Heatmap", fontsize=10)
        axes[1].axis("off")
        axes[2].imshow(superimposed)
        axes[2].set_title(f"Pred: {pred_class} [{correct}]", fontsize=10)
        axes[2].axis("off")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/gradcam_{i}_{correct}.png", dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  Sample {i+1}: {true_class} → {pred_class} [{correct}]")

    print(f"\nGrad-CAM images saved to {save_dir}/")