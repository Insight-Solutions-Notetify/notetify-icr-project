from django.db import models

# Create your models here.
class HandwritingImage(models.Model):
    image = models.ImageField(upload_to="uploads/")
    processed_image = models.ImageField(upload_to='processed/', blank=True, null=True)
    recognized_text = models.TextField(blank=True)
    processed = models.BooleanField(default=False)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    task_id = models.CharField(max_length=255, unique=True, null=True, blank=True)