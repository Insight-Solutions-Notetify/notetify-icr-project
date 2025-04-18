from django.urls import path
from .views import main, about, contact, register, login

urlpatterns = [
    path('', main, name='main'),
    path('about-the-team', about, name='about'),
    path('contact-us', contact, name='contact'),

    # Move this bottom two into authenticator app
    path('register', register, name='register'),
    path('login', login, name='login')
]