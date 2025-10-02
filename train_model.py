# train_model.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# === CONFIG ===
DATA_DIR = r"C:\Users\adith\OneDrive\Desktop\SignLanguageDetection\Data"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_INITIAL = 15
EPOCHS_FINE_TUNE = 10

print(f"TensorFlow version: {tf.__version__}")

# === DATA AUGMENTATION ===
# For TF 2.12: MobileNetV2 does NOT include preprocessing, so rescale=1./255 is correct
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.3,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    fill_mode='nearest'
)

train_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# === BUILD MODEL ===
# 🔥 TF 2.12: MobileNetV2 does NOT have include_preprocessing parameter
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
    # include_preprocessing is NOT available in TF 2.12
)

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(train_data.num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
print("✅ Model compiled. Starting initial training...")

# === INITIAL TRAINING ===
model.fit(train_data, validation_data=val_data, epochs=EPOCHS_INITIAL)

# === FINE-TUNING ===
print("\n🔄 Starting fine-tuning...")
base_model.trainable = True
# Freeze all layers except the last 50
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_data, validation_data=val_data, epochs=EPOCHS_FINE_TUNE)

# === SAVE ===
model.save("sign_language_model.h5")
print("\n✅ Model saved as 'sign_language_model.h5'")
print("Classes:", train_data.class_indices)