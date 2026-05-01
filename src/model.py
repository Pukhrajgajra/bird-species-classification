import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB3, ResNet50, MobileNetV2

NUM_CLASSES = 200
IMG_SIZE = (224, 224, 3)

def build_efficientnet(num_classes=NUM_CLASSES, trainable_base=False):
    base = EfficientNetB3(weights="imagenet", include_top=False, input_shape=IMG_SIZE)
    base.trainable = trainable_base
    inputs = tf.keras.Input(shape=IMG_SIZE)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = Model(inputs, outputs, name="EfficientNetB3")
    return model

def build_resnet50(num_classes=NUM_CLASSES, trainable_base=False):
    base = ResNet50(weights="imagenet", include_top=False, input_shape=IMG_SIZE)
    base.trainable = trainable_base
    inputs = tf.keras.Input(shape=IMG_SIZE)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = Model(inputs, outputs, name="ResNet50")
    return model

def build_mobilenet(num_classes=NUM_CLASSES, trainable_base=False):
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=IMG_SIZE)
    base.trainable = trainable_base
    inputs = tf.keras.Input(shape=IMG_SIZE)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = Model(inputs, outputs, name="MobileNetV2")
    return model

def get_model(name="efficientnet"):
    models = {
        "efficientnet": build_efficientnet,
        "resnet50":     build_resnet50,
        "mobilenet":    build_mobilenet,
    }
    return models[name]()

if __name__ == "__main__":
    for name in ["efficientnet", "resnet50", "mobilenet"]:
        model = get_model(name)
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        print(f"\n{name}: {model.count_params():,} parameters")
        print(f"Output shape: {model.output_shape}")