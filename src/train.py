import os
import sys
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping,
    ReduceLROnPlateau, TensorBoard, CSVLogger
)
import numpy as np
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_dataset_info, create_data_generators
from src.model import get_model

MODELS_DIR  = os.path.expanduser("~/bird_classification/models")
RESULTS_DIR = os.path.expanduser("~/bird_classification/results")
EPOCHS      = 30
FINE_TUNE_EPOCHS = 10

def train_model(model_name="efficientnet"):
    print(f"\n{'='*50}")
    print(f"Training: {model_name.upper()}")
    print(f"{'='*50}")

    train_df, test_df, classes_df = load_dataset_info()
    train_gen, val_gen, test_gen = create_data_generators(train_df, test_df)

    model = get_model(model_name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_accuracy")]
    )

    model.summary()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(f"{RESULTS_DIR}/logs/{model_name}", exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            filepath=f"{MODELS_DIR}/{model_name}_best.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        TensorBoard(
            log_dir=f"{RESULTS_DIR}/logs/{model_name}",
            histogram_freq=1
        ),
        CSVLogger(
            filename=f"{RESULTS_DIR}/{model_name}_training_log.csv",
            append=True
        )
    ]

    print(f"\nPhase 1: Training classifier head ({EPOCHS} epochs)")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\nPhase 2: Fine-tuning entire model ({FINE_TUNE_EPOCHS} epochs)")
    model.layers[1].trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_accuracy")]
    )

    fine_tune_history = model.fit(
        train_gen,
        epochs=FINE_TUNE_EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\nEvaluating on test set...")
    test_results = model.evaluate(test_gen, verbose=1)
    print(f"Test Accuracy:      {test_results[1]:.4f}")
    print(f"Top-5 Accuracy:     {test_results[2]:.4f}")

    results = {
        "model":          model_name,
        "test_accuracy":  float(test_results[1]),
        "top5_accuracy":  float(test_results[2]),
        "total_params":   model.count_params(),
        "best_val_acc":   float(max(history.history["val_accuracy"]))
    }

    with open(f"{RESULTS_DIR}/{model_name}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}/{model_name}_results.json")
    return model, history, results

if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "efficientnet"
    train_model(model_name)