# ============================================================
#  Sport Image Classification - Phase 2
#  Feature Extraction WITH Data Augmentation (MobileNetV2)
# ============================================================
#
# Approach (from the book, p. 231-233):
#   Unlike Phase 1 (which ran conv_base once offline and saved
#   NumPy arrays), here we build a single end-to-end Keras model:
#
#       Input -> Data Augmentation -> Frozen MobileNetV2 -> Dense classifier
#
#   Because every image passes through the conv base at each epoch,
#   training is much slower than Phase 1, but data augmentation can
#   be applied on the fly, which significantly reduces overfitting.
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
EPOCHS      = 50           # More epochs because augmentation delays overfitting
NUM_CLASSES = 8
SAVE_PATH   = '/content/drive/MyDrive/thesis/SportCNN/feature_extraction_with_augmentation.keras'

# -- 3. Build tf.data datasets (train 80 % / val 20 %) ------
#
#   validation_split + subset='training'/'validation' reproduces the 80 / 20 split used in Phase 1.

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

# -- 4. Data Augmentation stage -----
#
#   Keras augmentation layers are INACTIVE during evaluation/inference, so to never corrupt validation or test predictions.
#   (Book 8.14, p. 221)

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),   # 50 % horizontal mirror
        layers.RandomRotation(0.1),        # +-10 % of full circle (+-36 degrees)
        layers.RandomZoom(0.2),            # zoom in/out +-20 %
    ],
    name="data_augmentation"
)

# -- 5. Build the model -----
#
#   The model chains:
#     (a) Data augmentation
#     (b) MobileNetV2 preprocessing  (scales pixels to [-1, 1])
#     (c) Frozen MobileNetV2 conv base (include_top=False)
#     (d) Global Average Pooling  ->  Dense 256  ->  Dropout 0.5  ->  Dense 8
#
#   (Book 8.25, p. 232 - adapted for MobileNetV2 and 8 classes)

# 5a. Load the pretrained convolutional base (no top classifier)
conv_base = keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(*IMAGE_SIZE, 3)
)

# 5b. Freeze the conv base so only the Dense layers are trained
conv_base.trainable = False
print(f"Trainable weights after freezing: {len(conv_base.trainable_weights)}")  # -> 0

# 5c. Assemble the full model with the Functional API
inputs = keras.Input(shape=(*IMAGE_SIZE, 3))

# Apply data augmentation (active only during training)
x = data_augmentation(inputs)

# MobileNetV2-specific preprocessing: [0, 255] -> [-1, 1]
x = keras.applications.mobilenet_v2.preprocess_input(x)

# Pass through the frozen conv base (training=False keeps BatchNorm in inference mode, which is critical for a frozen base)
x = conv_base(x, training=False)

# Classifier head
x = layers.GlobalAveragePooling2D()(x)   # reduces (batch, h, w, c) -> (batch, c)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = keras.Model(inputs, outputs, name='MobileNetV2_Phase2_DataAug')
model.summary()

# -- 6. Compile -----
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.RMSprop(),  # default lr=1e-3
    metrics=['accuracy']
)

# -- 7. Train -----
#
#   ModelCheckpoint saves the best model (lowest val_loss).
#   Thanks to augmentation train for 50 epochs; overfitting is delayed compared to Phase 1. (Book p. 232-233)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=SAVE_PATH,
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    )
]

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=callbacks
)

# -- 8. Plot training curves -----
acc     = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss    = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc,     'bo', label='Training accuracy')
plt.plot(epochs_range, val_acc, 'b',  label='Validation accuracy')
plt.title('Training and Validation Accuracy\n(Phase 2 - Feature Extraction + Data Augmentation)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss,     'bo', label='Training loss')
plt.plot(epochs_range, val_loss, 'b',  label='Validation loss')
plt.title('Training and Validation Loss\n(Phase 2 - Feature Extraction + Data Augmentation)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('/content/drive/MyDrive/thesis/SportCNN/phase2_curves.png', dpi=150)
plt.show()
print("Training curves saved.")

# -- 9. Evaluate the best saved model -----
best_model = keras.models.load_model(SAVE_PATH)

val_loss_final, val_acc_final = best_model.evaluate(val_dataset)
print(f"\nBest model – Validation loss    : {val_loss_final:.4f}")
print(f"Best model – Validation accuracy: {val_acc_final:.4f}")

# -- 10. Save training log to CSV -----

df_history = pd.DataFrame({
    "epoch"      : range(1, EPOCHS + 1),
    "acc/train"  : history.history['accuracy'],
    "acc/valid"  : history.history['val_accuracy'],
    "loss/train" : history.history['loss'],
    "loss/valid" : history.history['val_loss'],
})
csv_path = '/content/drive/MyDrive/thesis/SportCNN/phase2_history.csv'
df_history.to_csv(csv_path, index=False)
print(f"\nTraining history saved to {csv_path}")
print(df_history.to_string(index=False))

# -- 11. (Added for curiosity, to remove in final report) Visualise augmented samples -----
plt.figure(figsize=(10, 10))
for images, _ in train_dataset.take(1):       # one batch from the dataset
    for i in range(9):
        augmented = data_augmentation(images)  # apply augmentation to the batch
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented[0].numpy().astype('uint8'))
        plt.axis('off')
plt.suptitle('Nine augmented versions of the same training image', y=1.02)
plt.tight_layout()
plt.show()