from django.shortcuts import render

# Create your views here.
def main(request):
    return render(request, 'main.html')

def about(request):
    return render(request, 'about.html')

def contact(request):
    return render(request, 'contact.html')

def register(request):
    return render(request, 'temp-create-acc.html')

def login(request):
    return render(request, 'temp-sign-acc.html')