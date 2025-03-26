from django.urls import path
from .views import HandwritingRecognitionView

urlpatterns = [
    path('recognize/', HandwritingRecognitionView.as_view(), name='recognize')
]