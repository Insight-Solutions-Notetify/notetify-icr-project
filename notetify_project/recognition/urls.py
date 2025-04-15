from django.urls import path
from .views import UploadImageView, ImageStatusView, upload_page, upload_image, result_view

urlpatterns = [
    # Separated and modularized methods (NEW)
    path('only_upload/', upload_image, name='upload_image'),
    path('result/<uuid:image_id>/', result_view, name='result_view'),

    # Old methods
    path('upload/', upload_page, name='upload_page'),
    path('api/upload/', UploadImageView.as_view(), name='upload_image'),
    path('api/status/<uuid:task_id>/', ImageStatusView.as_view(), name='image_status'),
]