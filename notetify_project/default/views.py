from django.shortcuts import render, redirect
from .forms import ContactForm
from django.core.mail import send_mail

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

            # Send an email
            send_mail(
                subject,
                message,
                email,
                ['to@example.com'],
                fail_silently=False,
            )

            return redirect('success')
        else:
            return render(request, 'contact.html', {'form': form})
    else:
        form = ContactForm()
    return render(request, 'contact.html', {'form': form})


def success(request):
    return render(request, 'success.html', {'message': 'Thank you for contacting us!'})