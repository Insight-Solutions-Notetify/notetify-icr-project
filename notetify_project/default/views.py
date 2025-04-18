from django.shortcuts import render

# Create your views here.
def main(request):
    return render(request, 'main.html')

def about(request):
    return render(request, 'about.html')

def contact(request):
    return render(request, 'contact.html')

def files(request):
    return render(request, 'temp-files.html')

def settings(request):
    return render(request, 'temp-settings.html')