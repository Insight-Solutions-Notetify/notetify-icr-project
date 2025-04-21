from django import forms
from django.core.validators import EmailValidator, RegexValidator, MinLengthValidator

class ContactForm(forms.Form):
    name = forms.CharField(max_length=100, validators=[MinLengthValidator(2)], widget=forms.TextInput(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500'}))
    email = forms.EmailField(validators=[EmailValidator(message="Enter a valid email address.")], widget=forms.TextInput(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500'}))
    phone = forms.CharField(max_length=15, validators=[
        RegexValidator(regex=r'^\+?1?\d{9,15}$', message="Phone number must be entered in the format: '+999999999'. Up to 15 digits allowed.")
    ], required=False, widget=forms.TextInput(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500'}))  # Optional field
    subject = forms.CharField(max_length=100, widget=forms.TextInput(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500'}))
    message = forms.CharField(widget=forms.Textarea(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500'}), validators=[MinLengthValidator(10)])