from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from .models import HandwritingImage
from .serializers import HandwritingImageSerializer
from .tasks import process_upload_image

def upload_page(request):
    return render(request, 'upload.html')

class UploadImageView(APIView):
    def post(self, request):
        serializer = HandwritingImageSerializer(data=request.data)
        if serializer.is_valid():
            instance = serializer.save()
            task = process_upload_image.delay(str(instance.id)) # Queue task
            instance.task_id = task.id
            instance.save()
            return Response({'id': str(task.id)}, status=status.HTTP_202_ACCEPTED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class ImageStatusView(APIView):
    def get(self, request, task_id):
        print("test")
        try:
            obj = HandwritingImage.objects.get(task_id=task_id)
            return Response({
                'processed': obj.processed,
                'text': obj.recognized_text,
                'processed_url': obj.processed_image.url if obj.processed_image else None
            })
        except HandwritingImage.DoesNotExist:
            return Response({'error': 'Not found'}, status=404)