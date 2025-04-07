import cv2
import numpy as np
from keras import models

from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.shortcuts import render
from django.conf import settings

from rest_framework.response import Response
from rest_framework import status, views
# from .models import HandwritingImage
from .serializers import HandwritingImageSerializer

# Import preprocessing and segmentation
from ml_modules import preprocessImage, segmentImage

# Create your views here.
model = models.load_model("model/emnist_model.keras")
# model.load_weights("model/CNN_model_StratifiedKFold_1_EMNIST.h5")
# model = models.load_model("model/CNN_model_StratifiedKFold_1_EMNIST.h5")
model.load_weights("model/emnist_model_loss0.68.weights.h5")

def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES["image"]
        image_path = default_storage.save(image.name, image)
        # print(image_path)
        uploaded_image = cv2.imread(f"media/{image_path}")
    
        # Send to preprocess and segmentation modules
        preprocessed = preprocessImage(uploaded_image)
        segmented, char_metadata = segmentImage(preprocessed)

        # Process predictions of character images
        predictions = model.predict(segmented)
        predicted_text = decode_prediction(predictions)

        # Attach predictions to metadata
        for i, pred in enumerate(predicted_text):
            char_metadata[i]['prediction'] = pred

        # Reassmble the sentence using metadata
        sorted_chars = sorted(char_metadata, key=lambda x: (x['line_idx'], x['word_idx'], x['char_idx']))

        text = ""
        cur_line, cur_word = -1, -1

        for char in sorted_chars:
            if char['line_idx'] != cur_line:
                text += "<br />"
                cur_line = char['line_idx']
                cur_word = -1
            if char['word_idx'] != cur_word:
                text += " "
                cur_word = char['word_idx']
            text += char['prediction']

        # Save Preprocessed Image (Remove later)
        success, img_buffer = cv2.imencode(".jpg", preprocessed)
        if not success:
            raise Exception("Failed to encode image")
        image_file = ContentFile(img_buffer.tobytes(), name='processed_image.jpg')
        processed_name = f"procsesed-{image_path}"
        default_storage.save(processed_name, image_file)

        # Output only the converted text and no images other than the file export
        image_url = f"{settings.MEDIA_URL}{image.name}"
        processed_url = f"{settings.MEDIA_URL}{processed_name}"
        predicted_text = text
        return render(request, 'upload.html', {'image_url':image_url, 'recognized_text':predicted_text, 'processed_url':processed_url})
    
    return render(request, 'upload.html')

def decode_prediction(predictions):
    # Convert model output into readable text (a-zA-Z & 0-9) Certain confidence threshold to be accepted else 'white_space' filler
    character_by_index = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                      'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                      'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    
    printed_guess = [character_by_index[np.argmax(ix)] for ix in predictions]
    # printed_guess = [np.argmax(ix) for ix in predictions]
    # print(printed_guess)
    confidence = [np.max(ix) for ix in predictions]

    for ix in range(len(printed_guess)):
        if confidence[ix] <= 0.70:
            printed_guess[ix] = ' '
    
    return printed_guess

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

            default_storage.save("processed-" + image.name, image_file)
            predicted_text = "Predicted text..."
            return Response({"text": predicted_text, "path": image_path}, status=status.HTTP_200_OK) 
        
            # segmented will be a list of lines, which is a list of words, which is a list of char images
            # Need to deconstruct down to a list of char images by line to processed

            predictions = model.predict(segmented)
            predicted_text = decode_prediction(predictions)
            return Response({"text": predicted_text}, status=status.HTTP_200_OK)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
