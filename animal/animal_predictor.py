import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from django.shortcuts import render
from django.core.files.storage import default_storage
import matplotlib.pyplot as plt
import io
import base64
from django.shortcuts import render
import gdown
import os
import requests
from tensorflow.keras.models import load_model

# Đường dẫn tới thư mục lưu trữ model trong dự án
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'models/animal_detection_model_weights.h5')

# URL của model trên S3
model_url = "https://predictionanimal.s3.us-east-1.amazonaws.com/animal_detection_model_weights.h5"  # Đúng URL của mô hình

# Tải model nếu chưa tồn tại trên server
if not os.path.exists(model_path):
    print("Downloading model from S3...")
    response = requests.get(model_url, stream=True)
    with open(model_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Model downloaded successfully.")

# Load model từ file đã tải xuống
model = load_model(model_path)


# Khởi tạo lại mô hình và nạp trọng số đã lưu

def build_model() -> tf.keras.Model:
    IMAGE_SIZE = 600
    NUM_CLASSES = 80  # Số lượng lớp trong mô hình của bạn

    inputs = tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    img_augmentation = tf.keras.models.Sequential([
        tf.keras.layers.RandomRotation(factor=0.15),
        tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        tf.keras.layers.RandomFlip(),
        tf.keras.layers.RandomContrast(factor=0.1),
    ])

    x = img_augmentation(inputs)

    base_model = tf.keras.applications.EfficientNetB7(
        include_top=False,
        input_tensor=x,
        weights="imagenet",
    )

    base_model.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model

def load_model():
    model = build_model()
    model.load_weights("animal_detection_model_weights.h5")
    return model

def preprocess_image(img_path, target_size=(600, 600)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

def predict_animal1(model, img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=-1)[0]
    confidence = np.max(predictions) * 100  # Convert confidence to percentage
    class_label = [name for name, idx in class_indices.items() if idx == predicted_class][0]
    return class_label, confidence  # Return both class_label and confidence as a percentage

model = load_model()
class_indices = {
    'Bear': 0, 'Brown bear': 1, 'Bull': 2, 'Butterfly': 3, 'Camel': 4, 'Canary': 5, 'Caterpillar': 6,
    'Cattle': 7, 'Centipede': 8, 'Cheetah': 9, 'Chicken': 10, 'Crab': 11, 'Crocodile': 12, 'Deer': 13,
    'Duck': 14, 'Eagle': 15, 'Elephant': 16, 'Fish': 17, 'Fox': 18, 'Frog': 19, 'Giraffe': 20, 'Goat': 21,
    'Goldfish': 22, 'Goose': 23, 'Hamster': 24, 'Harbor seal': 25, 'Hedgehog': 26, 'Hippopotamus': 27,
    'Horse': 28, 'Jaguar': 29, 'Jellyfish': 30, 'Kangaroo': 31, 'Koala': 32, 'Ladybug': 33, 'Leopard': 34,
'Lion': 35, 'Lizard': 36, 'Lynx': 37, 'Magpie': 38, 'Monkey': 39, 'Moths and butterflies': 40,
    'Mouse': 41, 'Mule': 42, 'Ostrich': 43, 'Otter': 44, 'Owl': 45, 'Panda': 46, 'Parrot': 47,
    'Penguin': 48, 'Pig': 49, 'Polar bear': 50, 'Rabbit': 51, 'Raccoon': 52, 'Raven': 53, 'Red panda': 54,
    'Rhinoceros': 55, 'Scorpion': 56, 'Sea lion': 57, 'Sea turtle': 58, 'Seahorse': 59, 'Shark': 60,
    'Sheep': 61, 'Shrimp': 62, 'Snail': 63, 'Snake': 64, 'Sparrow': 65, 'Spider': 66, 'Squid': 67,
    'Squirrel': 68, 'Starfish': 69, 'Swan': 70, 'Tick': 71, 'Tiger': 72, 'Tortoise': 73, 'Turkey': 74,
    'Turtle': 75, 'Whale': 76, 'Woodpecker': 77, 'Worm': 78, 'Zebra': 79
}
