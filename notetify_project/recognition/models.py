from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class HandwritingImage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="images")
    image = models.ImageField(upload_to="uploads/")
    processed_image = models.ImageField(upload_to='processed/', blank=True, null=True)
    recognized_text = models.TextField(blank=True)
    processed = models.BooleanField(default=False)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    task_id = models.CharField(max_length=255, unique=True, null=True, blank=True)