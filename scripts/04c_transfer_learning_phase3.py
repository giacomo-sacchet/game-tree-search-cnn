# ============================================================
#  Sport Image Classification - Phase 3
#  Fine-Tuning (MobileNetV2)
# ============================================================
#
# Approach (book p. 234-237):
#   Fine-tuning builds directly on Phase 2 (feature extraction with
#   data augmentation). The key idea is to unfreeze the top layers of
#   the already-trained convolutional base and jointly re-train them
#   together with our Dense classifier, using a very low learning rate
#
#   This "slightly adjusts" the high-level feature representations so
#   they become more relevant for our 8-class sport problem
#
#   Steps (book p. 234):
#     1. Add custom network on top of the pre-trained base.            done in Phase 2
#     2. Freeze the base network.                                      done in Phase 2
#     3. Train the part added.                                         done in Phase 2
#     4. Unfreeze the top layers of the base network.                  Phase 3
#     5. Jointly train both the unfrozen layers and the added part.    Phase 3
#
#   Note: Batch Normalization layers inside MobileNetV2 will be kept
#   in inference mode (trainable=False at the layer level) during
#   fine-tuning to avoid corrupting the running statistics learned on
#   ImageNet with our small-dataset mini-batch statistics
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
DATA_DIR          = '/content/Dataset_sport/Dataset'
IMAGE_SIZE        = (224, 224)   # MobileNetV2 native resolution
BATCH_SIZE        = 32
EPOCHS_FEATURE    = 50           # Phase 2 re-training (feature extraction with augmentation)
EPOCHS_FINETUNE   = 30           # Phase 3 fine-tuning epochs (book p. 236)
NUM_CLASSES       = 8

# Paths to save models on Google Drive
SAVE_PATH_PHASE2  = '/content/drive/MyDrive/thesis/SportCNN/feature_extraction_with_augmentation.keras'
SAVE_PATH_PHASE3  = '/content/drive/MyDrive/thesis/SportCNN/fine_tuning.keras'

# -- 3. Build tf.data datasets (train 80 % / val 20 %) -----
#
#   Same split and seed as Phase 1 and Phase 2

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
#   Identical to Phase 2: augmentation layers are INACTIVE during evaluation / inference. (Book 8.14, p. 221)

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),   # 50 % horizontal mirror
        layers.RandomRotation(0.1),        # +-10 % of full circle (+-36 degrees)
        layers.RandomZoom(0.2),            # zoom in/out +-20 %
    ],
    name="data_augmentation"
)

# -- 5. Build the Phase 2 model (feature extraction with augmentation) -----
#
#   Re-build the same model from Phase 2 with the conv base FROZEN.
#   This is mandatory before fine-tuning: the classifier must already be trained before unfreeze any conv layers, otherwise the large
#   random-weight gradients would destroy the pre-trained representations.
#   (Book p. 234 step 1-3)

# 5a. Load the pretrained convolutional base (no top classifier)
conv_base = keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(*IMAGE_SIZE, 3)
)

# 5b. Freeze the entire conv base for Phase 2 training
conv_base.trainable = False
print(f"Trainable weights after freezing: {len(conv_base.trainable_weights)}")  # → 0

# 5c. Assemble the full model with the Functional API (same as Phase 2)
inputs = keras.Input(shape=(*IMAGE_SIZE, 3))

x = data_augmentation(inputs)                                  # augmentation
x = keras.applications.mobilenet_v2.preprocess_input(x)        # scale [0,255] -> [-1,1]
x = conv_base(x, training=False)                               # frozen conv base

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = keras.Model(inputs, outputs, name='MobileNetV2_Phase3_FineTuning')
model.summary()

# -- 6. Phase 2 training: train only the Dense classifier -----
#
#   Use the best model saved by Phase 2 if it already exists on Drive, otherwise re-run the feature-extraction training from scratch and
#   save the checkpoint before moving on to fine-tuning
#   (Book 8.25 - 8.26, p. 232-233)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.RMSprop(),  # default lr = 1e-3
    metrics=['accuracy']
)

callbacks_phase2 = [
    keras.callbacks.ModelCheckpoint(
        filepath=SAVE_PATH_PHASE2,
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    )
]

print("\n=== Phase 2 – Feature extraction with data augmentation ===")
history_phase2 = model.fit(
    train_dataset,
    epochs=EPOCHS_FEATURE,
    validation_data=val_dataset,
    callbacks=callbacks_phase2
)

# Reload the best Phase 2 checkpoint before fine-tuning
model = keras.models.load_model(SAVE_PATH_PHASE2)
print(f"\nBest Phase 2 model loaded from: {SAVE_PATH_PHASE2}")

# -- 7. Unfreeze the top layers of MobileNetV2 for fine-tuning -----
#
#   MobileNetV2 is organised as a series of "blocks". Unfreeze only the last few layers (from layer index -30 onward), keeping the
#   lower, more generic layers frozen. This limits the risk of overfitting and preserves the low-level feature representations
#
#   (Book 8.27, p. 236 - adapted for MobileNetV2)

# Retrieve the MobileNetV2 sub-model from the loaded model
conv_base = model.get_layer('mobilenetv2_1.00_224')

# Unfreeze the conv base
conv_base.trainable = True

# but keep all layers before the last 30 frozen
for layer in conv_base.layers[:-30]:
    layer.trainable = False

# Keep every BatchNormalization layer in inference mode regardless
for layer in conv_base.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

# Report trainable weight count after partial unfreezing
trainable_count = len(model.trainable_weights)
print(f"\nTrainable weight tensors after partial unfreezing: {trainable_count}")

# -- 8. Re-compile with a very low learning rate -----
#
#   A very small LR (1e-5) is critical: large updates would destroy the feature representations i want to fine-tune. (Book 8.28, p. 236)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-5),
    metrics=['accuracy']
)

# -- 9. Fine-tune: jointly train unfrozen conv layers + Dense classifier -----
#
#   Train for 30 more epochs. Because the learning rate is very low and the conv base is only partially unfrozen, overfitting is delayed
#   (Book 8.28, p. 236)

callbacks_phase3 = [
    keras.callbacks.ModelCheckpoint(
        filepath=SAVE_PATH_PHASE3,
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    )
]

print("\n=== Phase 3 - Fine-tuning ===")
history_phase3 = model.fit(
    train_dataset,
    epochs=EPOCHS_FINETUNE,
    validation_data=val_dataset,
    callbacks=callbacks_phase3
)

# -- 10. Plot Phase 2 training curves -----
acc_p2      = history_phase2.history['accuracy']
val_acc_p2  = history_phase2.history['val_accuracy']
loss_p2     = history_phase2.history['loss']
val_loss_p2 = history_phase2.history['val_loss']
epochs_p2   = range(1, len(acc_p2) + 1)

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_p2, acc_p2,     'bo', label='Training accuracy')
plt.plot(epochs_p2, val_acc_p2, 'b',  label='Validation accuracy')
plt.title('Training and Validation Accuracy\n(Phase 2 – Feature Extraction + Data Augmentation)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_p2, loss_p2,     'bo', label='Training loss')
plt.plot(epochs_p2, val_loss_p2, 'b',  label='Validation loss')
plt.title('Training and Validation Loss\n(Phase 2 – Feature Extraction + Data Augmentation)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('/content/drive/MyDrive/thesis/SportCNN/phase2_curves_from_phase3.png', dpi=150)
plt.show()
print("Phase 2 training curves saved.")

# -- 11. Plot Phase 3 fine-tuning curves -----
acc_p3      = history_phase3.history['accuracy']
val_acc_p3  = history_phase3.history['val_accuracy']
loss_p3     = history_phase3.history['loss']
val_loss_p3 = history_phase3.history['val_loss']
epochs_p3   = range(1, len(acc_p3) + 1)

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_p3, acc_p3,     'bo', label='Training accuracy')
plt.plot(epochs_p3, val_acc_p3, 'b',  label='Validation accuracy')
plt.title('Training and Validation Accuracy\n(Phase 3 – Fine-Tuning)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_p3, loss_p3,     'bo', label='Training loss')
plt.plot(epochs_p3, val_loss_p3, 'b',  label='Validation loss')
plt.title('Training and Validation Loss\n(Phase 3 – Fine-Tuning)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('/content/drive/MyDrive/thesis/SportCNN/phase3_curves.png', dpi=150)
plt.show()
print("Phase 3 training curves saved.")

# -- 12. Evaluate the best fine-tuned model -----
#
#   Reload the best checkpoint (lowest val_loss during fine-tuning) and evaluate it on the validation set. (Book 8.28, p. 236)

best_model = keras.models.load_model(SAVE_PATH_PHASE3)

val_loss_final, val_acc_final = best_model.evaluate(val_dataset)
print(f"\nBest fine-tuned model - Validation loss    : {val_loss_final:.4f}")
print(f"Best fine-tuned model - Validation accuracy: {val_acc_final:.4f}")

# -- 13. Save training logs to CSV -----
#
#   Two CSV files: one per phase

# Phase 2 CSV
df_phase2 = pd.DataFrame({
    "epoch"      : range(1, EPOCHS_FEATURE + 1),
    "acc/train"  : history_phase2.history['accuracy'],
    "acc/valid"  : history_phase2.history['val_accuracy'],
    "loss/train" : history_phase2.history['loss'],
    "loss/valid" : history_phase2.history['val_loss'],
})
csv_path_p2 = '/content/drive/MyDrive/thesis/SportCNN/phase2_history_from_phase3.csv'
df_phase2.to_csv(csv_path_p2, index=False)
print(f"\nPhase 2 training history saved to {csv_path_p2}")

# Phase 3 CSV
df_phase3 = pd.DataFrame({
    "epoch"      : range(1, EPOCHS_FINETUNE + 1),
    "acc/train"  : history_phase3.history['accuracy'],
    "acc/valid"  : history_phase3.history['val_accuracy'],
    "loss/train" : history_phase3.history['loss'],
    "loss/valid" : history_phase3.history['val_loss'],
})
csv_path_p3 = '/content/drive/MyDrive/thesis/SportCNN/phase3_history.csv'
df_phase3.to_csv(csv_path_p3, index=False)
print(f"Phase 3 training history saved to {csv_path_p3}")
print(df_phase3.to_string(index=False))