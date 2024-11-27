from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class Prediction(models.Model):
    # Liên kết dự đoán với người dùng (có thể để trống nếu người dùng không đăng nhập)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)

    # Tên của loài động vật dự đoán, ví dụ: 'dog', 'cat', 'elephant'
    animal_type = models.CharField(max_length=255)  # Tên con vật dự đoán

    # Xác suất dự đoán (giá trị giữa 0 và 1, ví dụ: 0.95)
    probability = models.FloatField()  # Độ chính xác của dự đoán

    # Ảnh được tải lên để thực hiện dự đoán, lưu vào thư mục 'predictions/'
    image = models.ImageField(upload_to='predictions/', null=True, blank=True)  # Đường dẫn lưu ảnh

    # Thời gian dự đoán (tự động điền khi tạo bản ghi)
    timestamp = models.DateTimeField(default=timezone.now)  # Thời gian upload

    def __str__(self):
        # Hiển thị tên loài và xác suất khi in bản ghi
        username = self.user.username if self.user else "Anonymous"
        return f"{username} - {self.animal_type} ({self.probability * 100:.2f}%)"
