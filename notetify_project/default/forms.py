from django import forms
from django.core.validators import EmailValidator, RegexValidator, MinLengthValidator

class ContactForm(forms.Form):
    name = forms.CharField(max_length=100, validators=[MinLengthValidator(2)])
    email = forms.EmailField(validators=[EmailValidator(message="Enter a valid email address.")])
    phone = forms.CharField(max_length=15, validators=[
        RegexValidator(regex=r'^\+?1?\d{9,15}$', message="Phone number must be entered in the format: '+999999999'. Up to 15 digits allowed.")
    ], required=False)  # Optional field
    subject = forms.CharField(max_length=100)
    message = forms.CharField(widget=forms.Textarea, validators=[MinLengthValidator(10)])