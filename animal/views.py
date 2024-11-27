import messages
import tensorflow as tf
from PIL import Image
from django.contrib.auth.decorators import login_required

from .animal_predictor import predict_animal1
from django.shortcuts import render, get_object_or_404
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
from django.contrib.auth import logout
from django.shortcuts import render, redirect
from io import BytesIO

import base64
from django.conf import settings
from django.contrib import messages


from .models import Prediction


def login_page(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)

            # Check if the user is an admin
            if user.is_superuser:
                # Redirect to admin page if user is admin
                return redirect('admin_page')  # Adjust to your actual URL name for admin page
            else:
                # Redirect to user dashboard or home for non-admins
                return redirect('/home')
        else:
            # Handle invalid login
            return render(request, 'login.html', {'error': 'Invalid username or password'})

    return render(request, 'login.html')
def logout_view(request):
    logout(request)
    return redirect('login')  # Redirect to a desired page after logout

def edit_user(request, user_id):
    user = get_object_or_404(User, id=user_id)
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')

        user.username = username
        user.email = email
        user.save()

        messages.success(request, "User updated successfully!")
        return redirect('list_users')

    return render(request, 'edit_user.html', {'user': user})

@login_required
def add_user(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')

        # Kiểm tra nếu tên người dùng đã tồn tại
        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists. Please choose a different username.")
        else:
            if username and email and password:
                User.objects.create_user(username=username, email=email, password=password)
                messages.success(request, "User added successfully!")
                return redirect('admin_page')  # Chuyển hướng về trang admin

    return render(request, 'add_user.html')

def signup_page(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        User.objects.create_user(username=username, password=password)
        return redirect('/login')
    return render(request,'signup.html')

def admin_page(request):
    # Ensure only admin users can access this page
    if not request.user.is_superuser:
        return redirect('/home')  # Redirect non-admins to home or another page

    return render(request, 'admin_page.html')

def list_users(request):
    users = User.objects.all()
    return render(request, 'admin_page.html', {'users': users})

def delete_user(request, user_id):
    user = get_object_or_404(User, id=user_id)
    user.delete()
    messages.success(request, "User deleted successfully!")
    return redirect('list_users')

def list_predictions(request):
    predictions = Prediction.objects.all()
    return render(request, 'admin_page.html', {'predictions': predictions})
def profile(request):
    # Fetch user's prediction history if logged in
    user_history = Prediction.objects.filter(user=request.user).order_by(
        '-timestamp') if request.user.is_authenticated else []

    # Render the template with user history
    return render(request, 'profile.html', {
        'user_history': user_history
    })

def home(request):
    return render(request,'home.html')

# Khởi tạo lại mô hình và nạp trọng số đã lư
# Example usage in Django view
from django.core.files.storage import default_storage
from django.shortcuts import render
import os

from .models import Prediction  # Thêm dòng này để import mô hình Prediction

from .models import Prediction  # Đảm bảo bạn đã import model Prediction

def predict(request):
    predicted_class = None
    confidence = None
    image_file = None

    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        img_path = default_storage.save(f'media/{image_file.name}', image_file)
        absolute_img_path = default_storage.path(img_path)

        # Gọi hàm dự đoán và nhận kết quả
        predicted_class, confidence = predict_animal1(model, absolute_img_path)

        # Lưu kết quả dự đoán vào database
        Prediction.objects.create(
            user=request.user if request.user.is_authenticated else None,
            animal_type=predicted_class,
            probability=confidence,
            image=image_file
        )

    # Lấy lịch sử dự đoán của người dùng hiện tại
    user_history = Prediction.objects.filter(user=request.user).order_by('-timestamp') if request.user.is_authenticated else []

    # Trả về template với kết quả dự đoán và lịch sử dự đoán
    return render(request, 'getImage.html', {
        'animal': predicted_class,
        'confidence': confidence,
        'image_url': f"{settings.MEDIA_URL}{image_file.name}" if image_file else None,
        'user_history': user_history
    })



from django.shortcuts import render
from django.http import JsonResponse
from tensorflow.keras.preprocessing import image
import numpy as np

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

model = load_model()
class_indices = {'Bear': 0, 'Brown bear': 1, 'Bull': 2, 'Butterfly': 3, 'Camel': 4, 'Canary': 5, 'Caterpillar': 6,
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
                 'Turtle': 75, 'Whale': 76, 'Woodpecker': 77, 'Worm': 78, 'Zebra': 79}

# Hàm xử lý khung hình từ video
def process_video_frame(request):
    if request.method == 'POST':
        frame_data = request.POST.get('frame_data')
        image_data = base64.b64decode(frame_data.split(',')[1])
        image = Image.open(BytesIO(image_data)).resize((600, 600))
        img_array = np.expand_dims(np.array(image), axis=0)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

        # Dự đoán
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=-1)[0]
        confidence = float(np.max(predictions) * 100)  # Ép kiểu sang float
        class_label = [name for name, idx in class_indices.items() if idx == predicted_class][0]

        return JsonResponse({'animal': class_label, 'confidence': confidence})
    return render(request, 'video.html')