from django.urls import path
from .views import main, about, contact

urlpatterns = [
    path('', main, name='main'),
    path('about-the-team', about, name='about'),
    path('contact-us', contact, name='contact'),
]