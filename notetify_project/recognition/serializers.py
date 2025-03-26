from rest_framework import serializers
from .models import HandwritingImage

class HandwritingImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = HandwritingImage
        fields = ['id', 'image', 'uploaded_at']