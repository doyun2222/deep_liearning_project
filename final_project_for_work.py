from google.colab import drive
drive.mount('/content/drive')

%pwd

%cd /content/drive/MyDrive/seoul_tech_hw_dried_leaf/

%ls -l

import os
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
import matplotlib.pyplot as plt
import cv2
import math
import tensorflow as tf
from keras.optimizers.schedules import CosineDecayRestarts
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, Dropout
from tensorflow.keras.layers import Add, Concatenate, ReLU, BatchNormalization, GlobalAveragePooling2D, Multiply, Reshape
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
from sklearn.manifold import TSNE
from tensorflow.keras.models import Model
import tqdm
Image.MAX_IMAGE_PIXELS = None

path_healthy = '/content/drive/MyDrive/seoul_tech_hw_dried_leaf/dataset/healthy'
path_disease = '/content/drive/MyDrive/seoul_tech_hw_dried_leaf/dataset/disease'

healty_train = path_healthy + '/train'
disease_train = path_disease + '/train'
healty_test = path_healthy + '/test'
disease_test = path_disease + '/test'

healty_train_list = glob.glob(healty_train + '/*.jpg')
disease_train_list = glob.glob(disease_train + '/*.jpg')
healty_test_list = glob.glob(healty_test + '/*.jpg')
disease_test_list = glob.glob(disease_test + '/*.jpg')

healty_img = Image.open(healty_train_list[0])
healty_img

# 모델이 잎의 구멍이나 찢어진 부분을 더 잘 학습할 수 있도록 sharpen효과 적용
# class Sharpen (alpha=(0.2, 0.5), lightness=(0.5, 1.0), method='kernel', kernel_size=5, sigma=1.0, always_apply=None, p=0.5)
sharpen_jitter = A.Compose([
    A.Sharpen(alpha=(0.2, 0.7), lightness=(0.5, 1.0), method='kernel', p=1.0)
])
healty_img_aug = sharpen_jitter(image=np.array(healty_img))['image']
healty_img_aug = Image.fromarray(healty_img_aug)
healty_img_aug

# horizontal_flip
horizontal_flip = A.Compose([
    A.HorizontalFlip(p=1.0),
])
healty_img_aug = horizontal_flip(image=np.array(healty_img))['image']
healty_img_aug = Image.fromarray(healty_img_aug)
healty_img_aug

# vertical_flip
vertical_flip = A.Compose([
    A.VerticalFlip(p=0.5),
])
healty_img_aug = vertical_flip(image=np.array(healty_img))['image']
healty_img_aug = Image.fromarray(healty_img_aug)
healty_img_aug

# shear and rotation
shear_and_rotation = A.Compose([
    A.Affine(shear={'x': (-10, 10), 'y': (-10, 10)}, rotate=(-30, 30), p=1.0)
])
healty_img_aug = shear_and_rotation(image=np.array(healty_img))['image']
plt.imshow(healty_img_aug)

# Zoom in/out(ShiftScaleRotate)
zoom_in = A.Compose([
    A.ShiftScaleRotate(shift_limit=0, scale_limit=0.2, rotate_limit=0, p=1.0)
])
healty_img_aug = zoom_in(image=np.array(healty_img))['image']
plt.imshow(healty_img_aug)

geometric_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0, scale_limit=0.2, rotate_limit=0, p=0.5),
    A.Affine(shear={'x': (-10, 10), 'y': (-10, 10)}, rotate=(-30, 30), p=0.5),
    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), method='kernel', p=0.5),
])
healty_img_aug = geometric_pipeline(image=np.array(healty_img))['image']
plt.imshow(healty_img_aug)

disease_image = Image.open(disease_train_list[0])

def apply_color_highlighting(image_dataset):
    image = np.array(image_dataset)

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    yellow_mask = cv2.inRange(hsv, (25, 50, 50), (35, 255, 255))

    black_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))

    green_mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))

    combined_mask = yellow_mask | black_mask | green_mask

    highlighted = cv2.bitwise_and(image, image, mask=combined_mask)

    return Image.fromarray(highlighted)

highlighted = apply_color_highlighting(disease_image)

plt.imshow(highlighted)

# class ColorJitter (brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.5, 0.5), always_apply=None, p=0.5):
color_jitter = A.Compose([
    A.ColorJitter(brightness=(0.7, 1.2), contrast=(0.5, 1.5), saturation=(0.7, 1.5), hue=(-0.05, 0.05), p=1.0)
])
healty_img_aug = color_jitter(image=np.array(healty_img))['image']
healty_img_aug = Image.fromarray(healty_img_aug)
healty_img_aug

# channel shuffle
channel_shuffle = A.Compose([
    A.ChannelShuffle(p=1.0)
])
healty_img_aug = channel_shuffle(image=np.array(healty_img))['image']
healty_img_aug = Image.fromarray(healty_img_aug)
healty_img_aug

color_pipeline = A.Compose([
    A.ColorJitter(brightness=(0.7, 1.2), contrast=(0.5, 1.5), saturation=(0.7, 1.5), hue=(-0.05, 0.05), p=1.0),
    A.ChannelShuffle(p=1.0)
])
healty_img_aug = color_pipeline(image=np.array(healty_img))['image']
healty_img_aug = Image.fromarray(healty_img_aug)
healty_img_aug

aug_pipeline_total = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0, scale_limit=0.2, rotate_limit=0, p=0.5),
    A.Sharpen(alpha=(0.2, 0.7), lightness=(0.5, 1.0), method='kernel', p=0.5),
    A.ColorJitter(brightness=(0.7, 1.2), contrast=(0.5, 1.5), saturation=(0.7, 1.5), hue=(-0.05, 0.05), p=0.5),
])

aug_image = aug_pipeline_total(image=np.array(healty_img))['image']
aug_image = Image.fromarray(aug_image)
aug_image

healty_augmented_path = '/content/drive/MyDrive/2024_2st_semester/final_project/dataset/augmented/healthy'
disease_augmented_path = '/content/drive/MyDrive/2024_2st_semester/final_project/dataset/augmented/disease'

for path in tqdm.tqdm(healty_train_list):
  img = Image.open(path)
  filename = path.split('/')[-1].split('.')[0]

  geometric_aug = geometric_pipeline(image=np.array(img))['image']
  geometric_aug = Image.fromarray(geometric_aug)
  geometric_aug.save(healty_augmented_path + '/' + filename + '_geometric.jpg')

  color_aug = color_pipeline(image=np.array(img))['image']
  color_aug = Image.fromarray(color_aug)
  color_aug.save(healty_augmented_path + '/' + filename + '_color.jpg')

  total_aug = aug_pipeline_total(image=np.array(img))['image']
  total_aug = Image.fromarray(total_aug)
  total_aug.save(healty_augmented_path + '/' + filename + '_total.jpg')

  highlighted = apply_color_highlighting(img)
  highlighted.save(healty_augmented_path + '/' + filename + '_highlighted.jpg')

for path in tqdm.tqdm(disease_train_list):
  img = Image.open(path)
  filename = path.split('/')[-1].split('.')[0]

  geometric_aug = geometric_pipeline(image=np.array(img))['image']
  geometric_aug = Image.fromarray(geometric_aug)
  geometric_aug.save(disease_augmented_path + '/' + filename + '_geometric.jpg')

  color_aug = color_pipeline(image=np.array(img))['image']
  color_aug = Image.fromarray(color_aug)
  color_aug.save(disease_augmented_path + '/' + filename + '_color.jpg')

  total_aug = aug_pipeline_total(image=np.array(img))['image']
  total_aug = Image.fromarray(total_aug)
  total_aug.save(disease_augmented_path + '/' + filename + '_total.jpg')

  highlighted = apply_color_highlighting(img)
  highlighted.save(disease_augmented_path + '/' + filename + '_highlighted.jpg')


print("length of training data: ", len(healty_train_list))
print("length of test data: ", len(healty_test_list))
print("length of training disease: ", len(disease_train_list))
print("length of test disease: ", len(disease_test_list))

healty_augmented_path = '/content/drive/MyDrive/seoul_tech_hw_dried_leaf/dataset/augmented/healthy'
disease_augmented_path = '/content/drive/MyDrive/seoul_tech_hw_dried_leaf/dataset/augmented/disease'

# read augmented train data
healty_augmented_list = glob.glob(healty_augmented_path + '/*.jpg')
disease_augmented_list = glob.glob(disease_augmented_path + '/*.jpg')

print("length of augmented healthy training dataset: ", len(healty_augmented_list))
print("length of autmented disease training dataset: ", len(disease_augmented_list))

healty_train_list = healty_train_list + healty_augmented_list
disease_train_list = disease_train_list + disease_augmented_list

print(len(healty_train_list))
print(len(disease_train_list))

# 0 -> healty, 1 -> disease
healty_train_label = [0] * len(healty_train_list)
disease_train_label = [1] * len(disease_train_list)
healty_test_label = [0] * len(healty_test_list)
disease_test_label = [1] * len(disease_test_list)

train_label = healty_train_label + disease_train_label
test_label = healty_test_label + disease_test_label
train_list = healty_train_list + disease_train_list
test_list = healty_test_list + disease_test_list

train_data = np.array(train_list)
test_data = np.array(test_list)
train_label = np.array(train_label)
test_label = np.array(test_label)

# minimum size of images
# size_list = []
# for path in train_data:
#   img = Image.open(path)
#   size_list.append(img.size)
# print("minimum size of training img: ", min(size_list))

# size_list = []
# for path in test_data:
#   img = Image.open(path)
#   size_list.append(img.size)
# print("minimum size of test img: ", min(size_list))

def image_decode(data, label):
  img = tf.io.read_file(data)
  img = tf.image.decode_jpeg(img, channels=3)
  img = tf.image.resize(img, [256, 256])
  img = img / 255.0

  return tf.cast(img, tf.float32), label

train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))
train_dataset = train_dataset.map(image_decode).batch(32)

test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_label))
test_dataset = test_dataset.map(image_decode).batch(32)

print(train_dataset.element_spec)
print(test_dataset.element_spec)

for image, label in train_dataset.take(1):
  image = image[0].numpy()
  plt.imshow(image)
  plt.show()
  print(label[0])

# Simple Convolution layer (Lenet, C-P-C-P-C ...)
input = tf.keras.layers.Input(shape=(256, 256, 3))
layer1 = tf.keras.layers.Conv2D(filters=8, kernel_size=(7, 7), padding='same', activation='relu')(input) # 256x256
maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(layer1) # 128x128
layer2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='same', activation='relu')(maxpool) # 128x128
maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(layer2) # 64x64
layer3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(maxpool) # 64x64
maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(layer3) # 32x32
layer4 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(maxpool) # 32x32
maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(layer4) # 16x16
layer5 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(maxpool) # 16x16
maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(layer5) # 8x8
flatten = tf.keras.layers.Flatten()(maxpool)


# Dense layer
dense1 = tf.keras.layers.Dense(units=512, activation='relu')(flatten)
dense2 = tf.keras.layers.Dense(units=256, activation='relu')(dense1)
dense3 = tf.keras.layers.Dense(units=128, activation='relu')(dense2)
output = tf.keras.layers.Dense(units=1, activation='sigmoid')(dense3)

model = tf.keras.Model(inputs=input, outputs=output)


model.summary()

model.compile(optimizer=Adam(learning_rate=0.0008), loss='binary_crossentropy', metrics=['accuracy'])
# early_stopping = EarlyStopping(monitor='val_loss', patience=13, restore_best_weights=True)

history = model.fit(train_dataset, epochs=40, validation_data=test_dataset, verbose=1)

res = model.evaluate(test_dataset)
print('test accuracy: ', res[1] * 100)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# Accuracy graph
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'b-', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r--', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss graph
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'b-', label='Training Loss')
plt.plot(epochs, val_loss, 'r--', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

first_decay_steps=94
lr_decayed_fn = (CosineDecayRestarts(initial_learning_rate=0.006,
                                     first_decay_steps=first_decay_steps, m_mul=0.98, t_mul=3.0))

# learning_rate scheduler
def scheduler(epoch, lr):
  if epoch < 30:
    return lr
  else:
    return lr * math.exp(-0.02)

lr_scheduler = LearningRateScheduler(scheduler, verbose=1)

# model checkpoint
checkpoint_filepath = '/content/drive/MyDrive/seoul_tech_hw_dried_leaf/modelSaved.keras'
model_checkpoint_callback = ModelCheckpoint(filepath= checkpoint_filepath, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# apply global-average-pooling
# apply inception layer
# apply residual connection
# apply BatchNormalization
# apply SE Block

def inception_module(x, filters_1x1, filters_3x3, filters_5x5):
  # 1x1 Convolution
  conv1x1 = Conv2D(filters_1x1[0], (1, 1), padding='same', activation='relu')(x)

  # 1x1 Convolution + 3x3 Convolution
  conv3x3 = Conv2D(filters_1x1[1], (1, 1), padding='same', activation='relu')(x)
  conv3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu')(conv3x3)

  # 1x1 Convolution + 5x5 Convolution
  conv5x5 = Conv2D(filters_1x1[2], (1, 1), padding='same', activation='relu')(x)
  conv5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu')(conv5x5)

  # MaxPooling + 1x1 Convolution
  maxpool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
  maxpool_conv = Conv2D(filters_1x1[3], (1, 1), padding='same', activation='relu')(maxpool)

  # Concatenate all paths
  output = Concatenate()([conv1x1, conv3x3, conv5x5, maxpool_conv])
  return output

def residual_module(inputs, filters, filters_inception, kernel_size=3, strides=1):
  # Shortcut connection
  shortcut = inputs

  layer = Conv2D(filters, kernel_size, strides=strides, padding='same')(inputs)
  layer = BatchNormalization()(layer)
  layer = ReLU()(layer)


  inception = inception_module(layer, **filters_inception)
  inception = BatchNormalization()(inception)

  # If the input shape does not match the output, adjust the shortcut
  if (strides != 1) or (inputs.shape[-1] != inception.shape[-1]):
    shortcut = Conv2D(inception.shape[-1], kernel_size=1, strides=strides, padding='same')(inputs)
    shortcut = BatchNormalization()(shortcut)

  layer = Add()([inception, shortcut])
  layer = BatchNormalization()(layer)
  layer = ReLU()(layer)

  return layer

def se_block(input_tensor, reduction=16):
    """Squeeze-and-Excitation Block"""
    # Squeeze: Global Average Pooling
    channels = input_tensor.shape[-1]
    squeeze = GlobalAveragePooling2D()(input_tensor)

    # Excitation: Fully Connected layers
    excitation = Dense(channels // reduction, activation='relu')(squeeze)
    excitation = Dense(channels, activation='sigmoid')(excitation)

    # Scale: Reshape and Multiply
    excitation = Reshape((1, 1, channels))(excitation)
    scaled_tensor = Multiply()([input_tensor, excitation])

    return scaled_tensor

# model
input = Input(shape=(256, 256, 3))
layer = Conv2D(filters=16, kernel_size=(7, 7), strides=(1,1), padding='same')(input)
layer = BatchNormalization()(layer)
layer = ReLU()(layer)

layer = Conv2D(filters=32, kernel_size=(7, 7), strides=(1,1), padding='same')(layer)
layer = BatchNormalization()(layer)
layer = ReLU()(layer)

filters_inception1 = {'filters_1x1': [8, 8, 16, 8], 'filters_3x3': 16, 'filters_5x5': 32} # 64
layer = residual_module(layer, 64, filters_inception1, kernel_size=5, strides=2) # 256 -> 128

filters_inception2 = {'filters_1x1': [8, 32, 16, 16], 'filters_3x3': 64, 'filters_5x5': 40} # 128
layer = residual_module(layer, 128, filters_inception2, kernel_size=3, strides=2) # 128 -> 64
layer_se = se_block(layer, reduction=4)

filters_inception3 = {'filters_1x1': [32, 48, 16, 16], 'filters_3x3': 168, 'filters_5x5': 40} # 256
layer = residual_module(layer_se, 256, filters_inception3, kernel_size=3)
layer_se = se_block(layer, reduction=4)

filters_inception4 = {'filters_1x1': [64, 96, 16, 32], 'filters_3x3': 376, 'filters_5x5': 40} # 512
layer = residual_module(layer_se, 512, filters_inception4, kernel_size=3, strides=1)
layer_se = se_block(layer, reduction=8)
maxpool = MaxPooling2D((2, 2), strides=(2, 2))(layer_se)

filters_inception5 = {'filters_1x1': [192, 96, 16, 32], 'filters_3x3': 512, 'filters_5x5': 32} # 768
layer = residual_module(maxpool, 768, filters_inception5, kernel_size=3, strides=1)
layer_se = se_block(layer, reduction=8)
maxpool = MaxPooling2D((2, 2), strides=(2, 2))(layer_se)

filters_inception6 = {'filters_1x1': [384, 128, 16, 32], 'filters_3x3': 576, 'filters_5x5': 32} # 1024
layer = residual_module(maxpool, 1024, filters_inception6, kernel_size=3, strides=1)

global_avg_pooling = GlobalAveragePooling2D(name='global_avg_pooling')(layer)
dense = Dense(768, activation='relu')(global_avg_pooling)
dropout = Dropout(0.1)(dense)
dense = Dense(384, activation='relu')(dropout)
dropout = Dropout(0.1)(dense)
dense = Dense(192, activation='relu')(dropout)
output = Dense(1, activation='sigmoid')(dense)

model = tf.keras.Model(inputs=input, outputs=output)

model.summary()

# adjust class weight
class_weights = {0: 1.0, 1: 2.0}

model.compile(optimizer=AdamW(learning_rate=lr_decayed_fn, weight_decay=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
hist = model.fit(train_dataset, epochs=170, validation_data=test_dataset, class_weight=class_weights ,callbacks=[model_checkpoint_callback], verbose=1)

model = tf.keras.models.load_model(checkpoint_filepath)
res = model.evaluate(test_dataset, verbose=1)
print(f"Test Accuracy: {res[1] * 100:.2f}%")

# predict
y_pred_probs = model.predict(test_dataset.map(lambda x, y: x))
y_pred = (y_pred_probs > 0.5).astype(int)
test_labels_array = np.concatenate([y for x, y in test_dataset], axis=0)

# confusion matrix
cm = confusion_matrix(test_labels_array, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Healthy", "Disease"], yticklabels=["Healthy", "Disease"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(test_labels_array, y_pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "k--", label="Random Guessing")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# t-SNE feature extract
feature_extractor = Model(inputs=model.input, outputs=model.get_layer('global_avg_pooling').output)
train_features = np.concatenate([feature_extractor.predict(train_dataset.map(lambda x, y: x))])
test_features = np.concatenate([feature_extractor.predict(test_dataset.map(lambda x, y: x))])

train_labels_array = np.concatenate([y for x, y in train_dataset], axis=0)

# t-SNE visualization function
def plot_tsne(features, labels, title):
    tsne = TSNE(n_components=2, random_state=77, perplexity=30, max_iter=1000)
    tsne_result = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))
    for label, color, marker in zip([0, 1], ['blue', 'red'], ['o', 'o']):  # 채워진 마커로 변경
        plt.scatter(
            tsne_result[labels == label, 0],
            tsne_result[labels == label, 1],
            c=color,
            label=f"Class {label}",
            alpha=0.5,
            marker=marker
        )
    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

# t-SNE visualization
plot_tsne(train_features, train_labels_array, "t-SNE Visualization (Train Data)")
plot_tsne(test_features, test_labels_array, "t-SNE Visualization (Test Data)")

tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
precision = precision_score(test_labels_array, y_pred)
recall = recall_score(test_labels_array, y_pred)
f1 = f1_score(test_labels_array, y_pred)

# ROC-AUC score
roc_auc = roc_auc_score(test_labels_array, y_pred_probs)

# print result
print(f"Accuracy: {accuracy_score(test_labels_array, y_pred):.2f}")
print(f"Sensitivity (Recall): {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"Precision: {precision:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"ROC-AUC Score: {roc_auc:.2f}")