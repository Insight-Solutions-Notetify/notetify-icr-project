import tensorflow as tf
from keras import models
from src.modules.preprocessing import preprocessImage
from src.modules.segmentation import
from django.core.files.storage import default_storage
from rest_framework.response import Response
from rest_framework import status, views
from .models import HandwritingImage
from .serializers import HandwritingImageSerializer
from django.shortcuts import render

# Create your views here.
model = models.load_model("model/emnist_mdoel.keras")
model.load_weights