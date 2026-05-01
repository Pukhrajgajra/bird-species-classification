import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score
)
import pandas as pd
import os
import json

def evaluate_model(model, test_gen, class_names, model_name="model", save_dir="results"):
    os.makedirs(f"{save_dir}/plots", exist_ok=True)
    os.makedirs(f"{save_dir}/metrics", exist_ok=True)

    print("Generating predictions...")
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    top5_acc = np.mean([
        1 if y_true[i] in np.argsort(y_pred_probs[i])[-5:]
        else 0 for i in range(len(y_true))
    ])

    print(f"\nTest Accuracy:     {acc:.4f}")
    print(f"Top-5 Accuracy:    {top5_acc:.4f}")
    print(f"F1 Macro:          {f1_macro:.4f}")
    print(f"F1 Weighted:       {f1_weighted:.4f}")

    report = classification_report(
        y_true, y_pred,
        target_names=[c.split(".")[-1] for c in class_names],
        output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"{save_dir}/metrics/{model_name}_classification_report.csv")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, cmap="Blues", xticklabels=False, yticklabels=False)
    plt.title(f"{model_name} Confusion Matrix (200 species)", fontsize=16)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/plots/{model_name}_confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Confusion matrix saved!")

    per_class = report_df.iloc[:-3][["precision","recall","f1-score","support"]]
    worst_10 = per_class.nsmallest(10, "f1-score")
    best_10  = per_class.nlargest(10, "f1-score")

    print("\nTop 10 BEST classified species:")
    print(best_10[["f1-score","support"]].to_string())
    print("\nTop 10 WORST classified species:")
    print(worst_10[["f1-score","support"]].to_string())

    plot_training_curves(save_dir, model_name)

    results = {
        "model": model_name,
        "test_accuracy": float(acc),
        "top5_accuracy": float(top5_acc),
        "f1_macro":      float(f1_macro),
        "f1_weighted":   float(f1_weighted)
    }
    with open(f"{save_dir}/metrics/{model_name}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results

def plot_training_curves(save_dir, model_name):
    csv_path = f"{save_dir}/{model_name}_training_log.csv"
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(df["accuracy"], label="Train")
    axes[0].plot(df["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy over epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(df["loss"], label="Train")
    axes[1].plot(df["val_loss"], label="Validation")
    axes[1].set_title("Loss over epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    plt.suptitle(f"{model_name} Training Curves", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/plots/{model_name}_training_curves.png", dpi=150)
    plt.close()
    print(f"Training curves saved!")