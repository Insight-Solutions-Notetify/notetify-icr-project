import tensorflow as tf
from keras import models
from src.modules.preprocessing import preprocessImage
from src.modules.segmentation import segmentate_image, segment_characters
from django.core.files.storage import default_storage
from rest_framework.response import Response
from rest_framework import status, views
from .models import HandwritingImage
from .serializers import HandwritingImageSerializer
from django.shortcuts import render

# Create your views here.
model = models.load_model("model/emnist_mdoel.keras")
model.load_weights("model/emnist_model_loss0.68.weights.h5")

class HandwritingRecognitionView(views.APIView):
    def post(self, request):
        serializer = HandwritingImageSerializer(data=request.data)
        if serializer.is_valid():
            image = serializer.validated_data["image"]
            image_path = default_storage.save(image.name, image)

            # Send to preprocess and segmentation modules
            preprocessed = preprocessImage(image)
            # segmented = segmentateImage(image)
            segmented = segment_characters(preprocessed) #TEMP while segementImage is being worked on
            # segmented will be a list of lines, which is a list of words, which is a list of char images
            # Need to deconstruct down to a list of char images by line to processed

            predictions = model.predict(segmented)
            predicted_text = decode_prediction(predictions)

            return Response({"text": predicted_text}, status=status.HTTP_200_OK)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
def decode_prediction(prediction):
    # Covner model output into readable text (a-zA-Z & 0-9) Certain confidence threshold to be accepted else 'white_space' filler

    return "Recognized Text Here"