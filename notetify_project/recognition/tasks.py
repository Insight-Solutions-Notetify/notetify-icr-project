import cv2
import numpy as np
from keras import models

from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from celery import shared_task
from .models import HandwritingImage
from ml_modules import preprocessImage, segmentImage

model = models.load_model('model/emnist_model.keras')
model.load_weights('model/emnist_model_loss0.68.weights.h5')

@shared_task
def process_upload_image(image_id):
    obj = HandwritingImage.objects.get(id=image_id)
    img = cv2.imread(obj.image.path)

    # Preprocess & segment
    preprocessed = preprocessImage(img)
    segmented, metadata = segmentImage(preprocessed)

    # Predict
    predictions = model.predict(segmented)
    predicted_text = decode_prediction(predictions)

    # Attach predictions
    for i, pred in enumerate(predicted_text):
        metadata[i]['prediction'] = pred

    sorted_chars = sorted(metadata, key=lambda x: (x['line_idx'], x['word_idx'], x['char_idx']))

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

    # Save preprocess image
    success, buffer = cv2.imencode('.jpg', preprocessed)
    if success:
        processed_file = ContentFile(buffer.tobytes(), name=f'processed-{obj.image.name.split("/")[-1]}')
        obj.processed_image.save(processed_file.name, processed_file, save=False)

    obj.recognized_text = text
    obj.processed = True
    obj.save()

def decode_prediction(predictions):
    character_by_index = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    printed_guess = [character_by_index[np.argmax(ix)] for ix in predictions]
    confidence = [np.max(ix) for ix in predictions]

    for ix in range(len(printed_guess)):
        if confidence[ix] <= 0.70:
            printed_guess[ix] = ' '
    
    return printed_guess