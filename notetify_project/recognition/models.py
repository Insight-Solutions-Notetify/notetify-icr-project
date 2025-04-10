from django.db import models

# Create your models here.
class HandwritingImage(models.Model):
    image = models.ImageField(upload_to="uploads/")
    recognized_text = models.TextField(blank=True)
    processed = models.BooleanField(default=False)
    uploaded_at = models.DateTimeField(auto_now_add=True)