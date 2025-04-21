from django.shortcuts import render, redirect
from .forms import ContactForm
from django.http import HttpResponse
from django.core.mail import EmailMessage
from django.conf import settings

# Create your views here.
def main(request):
    return render(request, 'main.html')

def about(request):
    return render(request, 'about.html')

def settings(request):
    return render(request, 'temp-settings.html')

def contact(request):
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            name = form.cleaned_data['name']
            email = form.cleaned_data['email']
            phone = form.cleaned_data['phone']
            subject = form.cleaned_data['subject']
            message = form.cleaned_data['message']

            # # Send an email
            # EmailMessage(
            #     'Contact Form Submission from {}'.format(name),
            #     message,
            #     'notetify@gamil.com', # Send from
            #     ['admin@gmail.com'], # Send to
            #     [],
            #     reply_to=[email] # Email from the form to get back to
            # ).send()

            return redirect('success')
        else:
            return render(request, 'contact.html', {'form': form})
    else:
        form = ContactForm()
    return render(request, 'contact.html', {'form': form})


def success(request):
    return render(request, 'success.html', {'message': 'Thank you for contacting us!'})