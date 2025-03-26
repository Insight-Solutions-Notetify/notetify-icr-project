import tensorflow as tf
import cv2
from keras import models
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from rest_framework.response import Response
from rest_framework import status, views
from .models import HandwritingImage
from .serializers import HandwritingImageSerializer
from django.shortcuts import render

# Import preprocessing and segmentation
from ml_modules.modules.preprocessing import preprocessImage
from ml_modules.modules.segmentation import segmentate_image, segment_characters

# Create your views here.
model = models.load_model("model/emnist_model.keras")
model.load_weights("model/emnist_model_loss0.68.weights.h5")

class HandwritingRecognitionView(views.APIView):
    def post(self, request):
        serializer = HandwritingImageSerializer(data=request.data)
        if serializer.is_valid():
            image = serializer.validated_data["image"]
            image_path = default_storage.save(image.name, image)

            uploaded_image = cv2.imread(f"media/{image_path}")
        
            # Send to preprocess and segmentation modules
            preprocessed = preprocessImage(uploaded_image)
            # segmented = segmentateImage(image)
            # segmented = segment_characters(preprocessed) #TEMP while segementImage is being worked on

            success, img_buffer = cv2.imencode(".jpg", preprocessed)
            if not success:
                raise Exception("Failed to encode image")
            image_file = ContentFile(img_buffer.tobytes(), name='processed_image.jpg')

            default_storage.save("Processed" + image.name, image_file)
            predicted_text = "Predicted text..."
            return Response({"text": predicted_text, "path": image_path}, status=status.HTTP_200_OK) 
        
            # segmented will be a list of lines, which is a list of words, which is a list of char images
            # Need to deconstruct down to a list of char images by line to processed

            predictions = model.predict(segmented)
            predicted_text = decode_prediction(predictions)
            return Response({"text": predicted_text}, status=status.HTTP_200_OK)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES["image"]
        image_path = default_storage.save(image.name, image)

        uploaded_image = cv2.imread(f"media/{image_path}")
    
        # Send to preprocess and segmentation modules
        preprocessed = preprocessImage(uploaded_image)
        # segmented = segmentateImage(image)
        # segmented = segment_characters(preprocessed) #TEMP while segementImage is being worked on

        success, img_buffer = cv2.imencode(".jpg", preprocessed)
        if not success:
            raise Exception("Failed to encode image")
        image_file = ContentFile(img_buffer.tobytes(), name='processed_image.jpg')

        processed_image = default_storage.save("Processed" + image.name, image_file)
        predicted_text = "Predicted text..."
        return render(request, 'upload.html', {'image_url': default_storage.url(image_path), 'recognized_text':predicted_text, 'processed_url':default_storage.url(processed_image)})
    
    return render(request, 'upload.html')

def decode_prediction(prediction):
    # Covner model output into readable text (a-zA-Z & 0-9) Certain confidence threshold to be accepted else 'white_space' filler

    return "Recognized Text Here"