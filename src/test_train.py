import os
import sys
import tensorflow as tf

sys.path.append(os.path.expanduser("~/bird_classification"))
from src.data_loader import load_dataset_info, create_data_generators
from src.model import get_model

train_df, test_df, classes_df = load_dataset_info()
train_gen, val_gen, test_gen = create_data_generators(train_df, test_df)

model = get_model("efficientnet")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_accuracy")]
)

print("Starting 3-epoch test run...")
history = model.fit(
    train_gen,
    epochs=3,
    validation_data=val_gen,
    verbose=1
)

print("Test run complete!")
print(f"Final val accuracy: {history.history['val_accuracy'][-1]:.4f}")