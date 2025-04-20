from django.urls import path
from .views import UploadImageView, ImageStatusView, upload_page, upload_image, result_view

urlpatterns = [
    # Separated and modularized methods (NEW)
    path('only_upload/', upload_image, name='upload'),
    path('result/<uuid:image_id>/', result_view, name='result_view'),

]