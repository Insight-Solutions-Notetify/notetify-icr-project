from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.http import HttpResponseForbidden
from .models import HandwritingImage
from .tasks import process_upload_image

@login_required
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

@login_required
def result_view(request, image_id):
    image = get_object_or_404(HandwritingImage, task_id=image_id)

    if image.user != request.user:
        return HttpResponseForbidden("You do not have permission to view this image.")
    
    return render(request, 'result.html', {
        'image': image,
        'task_id': image.task_id,
    })

@login_required
def my_uploads(request):
    images = HandwritingImage.objects.filter(user=request.user).order_by('-uploaded_at')
    paginator = Paginator(images, 5)

    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    return render(request, 'my_uploads.html', {'page_obj': page_obj})

@login_required
def delete_image(request, id):
    image = get_object_or_404(HandwritingImage, id=id, user=request.user)
    if request.method == "POST":
        image.delete()
    return redirect('my_uploads')

@login_required
def reprocess_image(request, id):
    image = get_object_or_404(HandwritingImage, id=id, user=request.user)
    if request.method == "POST":
        image.processed = False
        image.recognized_text = ""
        image.task_id = process_upload_image.delay(image.task_id)
        image.save()
    return redirect('my_uploads')