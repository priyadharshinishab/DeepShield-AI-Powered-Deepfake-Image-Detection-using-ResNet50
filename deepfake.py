import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import json

# -------------------------------
# ✅ Define paths (update here only)
# -------------------------------
BASE_DIR = r"C:\Users\Priyadharshini\deepfake_resnet_project\data"
REAL_IMG_DIR = os.path.join(BASE_DIR, "real_cifake_images-20251109T143758Z-1-001")
FAKE_IMG_DIR = os.path.join(BASE_DIR, "fake_cifake_images-20251109T143745Z-1-001")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test-20251109T143816Z-1-001", "test")

# Output JSON
OUTPUT_JSON = os.path.join(r"C:\Users\Priyadharshini\deepfake_resnet_project", "teamname_prediction.json")

# -------------------------------
# ✅ Check folders exist
# -------------------------------
for folder in [REAL_IMG_DIR, FAKE_IMG_DIR, TEST_IMG_DIR]:
    if not os.path.exists(folder):
        raise FileNotFoundError(f"❌ Missing folder: {folder}")
print("✅ All folders verified!")

# -------------------------------
# ✅ Prepare temporary dataset structure in memory
# -------------------------------
train_datagen = ImageDataGenerator(validation_split=0.2, rescale=1.0/255)

train_gen = train_datagen.flow_from_directory(
    BASE_DIR,
    classes=[
        "real_cifake_images-20251109T143758Z-1-001",
        "fake_cifake_images-20251109T143745Z-1-001"
    ],
    target_size=(224, 224),
    batch_size=8,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    BASE_DIR,
    classes=[
        "real_cifake_images-20251109T143758Z-1-001",
        "fake_cifake_images-20251109T143745Z-1-001"
    ],
    target_size=(224, 224),
    batch_size=8,
    class_mode="binary",
    subset="validation",
    shuffle=True
)

test_datagen = ImageDataGenerator(rescale=1.0/255)
test_gen = test_datagen.flow_from_directory(
    os.path.join(BASE_DIR, "test-20251109T143816Z-1-001"),
    target_size=(224, 224),
    batch_size=1,
    class_mode=None,
    shuffle=False
)

# -------------------------------
# ✅ Build Model (ResNet50)
# -------------------------------
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation="relu")(x)
preds = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=preds)

# Freeze base model
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# -------------------------------
# ✅ Train for demo (only few epochs)
# -------------------------------
print("\n🚀 Training on small subset for demo...\n")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=2,
    steps_per_epoch=5,
    validation_steps=2
)

# -------------------------------
# ✅ Predict on test images
# -------------------------------
print("\n🔍 Running predictions on test set...")
preds = model.predict(test_gen)
pred_labels = ["REAL" if p > 0.5 else "FAKE" for p in preds]

# Save predictions
results = {os.path.basename(path): label for path, label in zip(test_gen.filepaths, pred_labels)}

with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=4)

print(f"\n✅ Predictions saved to: {OUTPUT_JSON}")
print("🔹 Example output:", list(results.items())[:5])
