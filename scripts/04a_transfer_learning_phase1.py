# ============================================================
#  Sport Image Classification - Phase 1
#  Fast Feature Extraction WITHOUT Data Augmentation (MobileNetV2)
# ============================================================
#
# Approach (from the book, p. 229-231):
#   Unlike Phase 2 (which builds a single end-to-end model and passes
#   every image through the conv base at each epoch), here we run the
#   frozen MobileNetV2 base ONCE over the entire dataset and cache the
#   resulting feature maps as NumPy arrays. A small Dense classifier
#   is then trained on those cached features.
#
#       dataset -> MobileNetV2 (once, offline) -> NumPy arrays -> Dense classifier (trained)
#
#   Advantage : very fast training, only Dense layers are updated.
#   Limitation: data augmentation cannot be used (images processed once).
# ============================================================

# -- 0. Mount Google Drive and extract dataset -----
from google.colab import drive
drive.mount('/content/drive/', force_remount=True)

source = '/content/drive/MyDrive/thesis/SportCNN/Dataset_sport.zip'
dest   = '/content/Dataset_sport.zip'
!cp $source $dest

!mkdir -p /content/Dataset_sport
!unzip -q /content/Dataset_sport.zip -d /content/Dataset_sport
!ls /content/Dataset_sport

# -- 1. Imports -----
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory

# -- 2. Configuration -----
DATA_DIR    = '/content/Dataset_sport/Dataset'
IMAGE_SIZE  = (224, 224)   # MobileNetV2 native resolution
BATCH_SIZE  = 32
EPOCHS      = 20           # Fast: only Dense layers are trained
NUM_CLASSES = 8
SAVE_PATH   = '/content/drive/MyDrive/thesis/SportCNN/feature_extraction.keras'

# -- 3. Build tf.data datasets (train 80 % / val 20 %) -----
#
#   image_dataset_from_directory handles reading, decoding, resizing and batching automatically.
#   validation_split + subset='training'/'validation' 80 / 20 split

train_dataset = image_dataset_from_directory(
    DATA_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset='training',
    seed=42,
    label_mode='int'
)

val_dataset = image_dataset_from_directory(
    DATA_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset='validation',
    seed=42,
    label_mode='int'
)

class_names = train_dataset.class_names
print(f"Classes ({NUM_CLASSES}): {class_names}")

# Prefetch for performance (overlap GPU compute with CPU data loading)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset   = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# -- 4. Load the MobileNetV2 convolutional base (pretrained, no top) -----
#
#   include_top=False removes the original ImageNet classification head
#   The base is frozen: its weights will not be updated during training
#   (Book 8.23, p. 231)

conv_base = keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(*IMAGE_SIZE, 3)
)

conv_base.trainable = False
print(f"Trainable weights after freezing: {len(conv_base.trainable_weights)}")  # -> 0

# -- 5. Extract features (run dataset once through the frozen base) -----
#
#   MobileNetV2 expects pixels scaled to [-1, 1]: apply mobilenet_v2.preprocess_input before each forward pass
#   Features are cached as NumPy arrays so training only touches the Dense classifier. (Book 8.20, p. 229)

def get_features_and_labels(dataset):
    all_features = []
    all_labels   = []
    for images, labels in dataset:
        preprocessed = keras.applications.mobilenet_v2.preprocess_input(images)
        features      = conv_base.predict(preprocessed, verbose=0)
        all_features.append(features)
        all_labels.append(labels.numpy())
    return np.concatenate(all_features), np.concatenate(all_labels)

print("\nExtracting training features ...")
train_features, train_labels = get_features_and_labels(train_dataset)

print("Extracting validation features ...")
val_features, val_labels = get_features_and_labels(val_dataset)

print(f"\ntrain_features shape : {train_features.shape}")   # (N_train, 7, 7, 1280)
print(f"val_features   shape : {val_features.shape}")       # (N_val,   7, 7, 1280)

# -- 6. Build the Dense classifier on top of the extracted features -----
#
#   Input shape matches the MobileNetV2 output: (7, 7, 1280)
#   GlobalAveragePooling2D reduces it to (1280,) before the Dense head
#   (Book 8.21, p. 229  adapted for MobileNetV2 and 8 classes)

inputs  = keras.Input(shape=conv_base.output_shape[1:])   # (7, 7, 1280)
x       = layers.GlobalAveragePooling2D()(inputs)
x       = layers.Dense(256, activation='relu')(x)
x       = layers.Dropout(0.5)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = keras.Model(inputs, outputs, name='MobileNetV2_Phase1_NoAug')
model.summary()

# -- 7. Compile -----
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-4),
    metrics=['accuracy']
)

# -- 8. Train -----
#
#   Train directly on NumPy arrays (not a tf.data pipeline) because the features are already cached in memory.
#   ModelCheckpoint saves the best model (lowest val_loss).
#   (Book 8.21, p. 230)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=SAVE_PATH,
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    )
]

history = model.fit(
    train_features, train_labels,
    epochs=EPOCHS,
    validation_data=(val_features, val_labels),
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

# -- 9. Plot training curves -----
acc      = history.history['accuracy']
val_acc  = history.history['val_accuracy']
loss     = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc,     'bo', label='Training accuracy')
plt.plot(epochs_range, val_acc, 'b',  label='Validation accuracy')
plt.title('Training and Validation Accuracy\n(Phase 1 – Fast Feature Extraction)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss,     'bo', label='Training loss')
plt.plot(epochs_range, val_loss, 'b',  label='Validation loss')
plt.title('Training and Validation Loss\n(Phase 1 – Fast Feature Extraction)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('/content/drive/MyDrive/thesis/SportCNN/phase1_curves.png', dpi=150)
plt.show()
print("Training curves saved.")

# -- 10. Evaluate the best saved model -----
best_model = keras.models.load_model(SAVE_PATH)

val_loss_final, val_acc_final = best_model.evaluate(
    val_features, val_labels, verbose=0
)
print(f"\nBest model – Validation loss    : {val_loss_final:.4f}")
print(f"Best model – Validation accuracy: {val_acc_final:.4f}")

# -- 11. Save training log to CSV -----

df_history = pd.DataFrame({
    "epoch"      : range(1, EPOCHS + 1),
    "acc/train"  : history.history['accuracy'],
    "acc/valid"  : history.history['val_accuracy'],
    "loss/train" : history.history['loss'],
    "loss/valid" : history.history['val_loss'],
})
csv_path = '/content/drive/MyDrive/thesis/SportCNN/phase1_history.csv'
df_history.to_csv(csv_path, index=False)
print(f"\nTraining history saved to {csv_path}")
print(df_history.to_string(index=False))