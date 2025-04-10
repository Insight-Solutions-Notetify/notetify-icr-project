from django.urls import path
from .views import UploadImageView, ImageStatusView, upload_page

urlpatterns = [
    path('upload/', upload_page, name='upload_page'),
    path('api/upload/', UploadImageView.as_view(), name='upload_image'),
    path('api/status/<uuid:task_id>/', ImageStatusView.as_view(), name='image_status'),
]