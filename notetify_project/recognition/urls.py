from django.urls import path, re_path
from .views import upload_image, result_view, my_uploads, check_processing_status, delete_image, reprocess_image

urlpatterns = [
    # Separated and modularized methods (NEW)
    path('only_upload/', upload_image, name='upload'),
    path('result/<uuid:image_id>/', result_view, name='result_view'),

    path('my_uploads/', my_uploads, name = 'my_uploads'),
    path("api/check_status/", check_processing_status, name="check_processing_status"),
    path('image/<int:id>/delete/', delete_image, name='delete_image'),
    path('image/<int:id>/reprocess/', reprocess_image, name='reprocess_image'),
]