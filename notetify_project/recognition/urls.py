from django.urls import path
from .views import HandwritingRecognitionView, upload_image

urlpatterns = [
    path('recognize/', HandwritingRecognitionView.as_view(), name='recognize'),
    path('upload/', upload_image, name='upload_image'),
]

#  curl -X POST -F "image=@src/NCR_samples/top-flash-sample-2.jpg" http://127.0.0.1:8000/api/recognize/ # Usage for CURL to website