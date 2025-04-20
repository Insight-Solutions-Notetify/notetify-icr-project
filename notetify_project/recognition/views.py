from django.shortcuts import render, get_object_or_404, redirect
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from django.http import HttpResponseForbidden
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from .models import HandwritingImage
from .serializers import HandwritingImageSerializer
from .tasks import process_upload_image


def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        if request.user.is_authenticated:
            image_file = request.FILES['image']
            image_obj = HandwritingImage.objects.create(
                user=request.user,
                image=image_file
            )
            task = process_upload_image.delay(image_obj.id)
            image_obj.task_id = task.id
            image_obj.save()

        return render(request, 'upload_success.html', {'image_id': task.id})
    return render(request, 'upload_file.html')

def result_view(request, image_id):
    image = get_object_or_404(HandwritingImage, task_id=image_id)

    if image.user != request.user:
        return HttpResponseForbidden("You do not have permission to view this image.")
    
    return render(request, 'result.html', {
        'image': image,
        'task_id': image.task_id,
    })