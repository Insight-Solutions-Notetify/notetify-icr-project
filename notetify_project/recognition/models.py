from django.db import models
from django.contrib.auth.models import User

# Create your models here.

def upload_path(instance, filename):
    return f'user_{instance.user.id}/uploads/{filename}'

def processed_path(instance, filename):
    return f'user_{instance.user.id}/processed/{filename}'

class HandwritingImage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="images")
    image = models.ImageField(upload_to=upload_path)
    processed_image = models.ImageField(upload_to=processed_path, blank=True, null=True)
    recognized_text = models.TextField(blank=True)
    processed = models.BooleanField(default=False)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    task_id = models.CharField(max_length=255, unique=True, null=True, blank=True)