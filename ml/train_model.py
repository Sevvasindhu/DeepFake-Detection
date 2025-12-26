import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = 128
EPOCHS = 10
BATCH_SIZE = 32

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "backend", "deepfake_model.h5")


# -----------------------------
# LOAD DATA
# -----------------------------
def load_images(folder, label):
    images = []
    labels = []

    for file in os.listdir(folder):
        try:
            img_path = os.path.join(folder, file)
            img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
            img = img_to_array(img) / 255.0
            images.append(img)
            labels.append(label)
        except Exception as e:
            print(f"Skipping {file}: {e}")

    return images, labels


real_images, real_labels = load_images(os.path.join(DATASET_PATH, "real"), 0)
fake_images, fake_labels = load_images(os.path.join(DATASET_PATH, "fake"), 1)

X = np.array(real_images + fake_images)
y = np.array(real_labels + fake_labels)

print("Total images:", len(X))

# -----------------------------
# SPLIT DATA
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# BUILD CNN MODEL
# -----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# TRAIN MODEL
# -----------------------------
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2
)

# -----------------------------
# EVALUATE MODEL
# -----------------------------
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc * 100:.2f}%")

# -----------------------------
# SAVE FULL MODEL (IMPORTANT)
# -----------------------------
os.makedirs("backend", exist_ok=True)
model.save(MODEL_PATH)

print(f"✅ Model saved at {MODEL_PATH}")

# -----------------------------
# PLOT ACCURACY
# -----------------------------
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Model Accuracy")
plt.show()
