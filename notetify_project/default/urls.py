from django.urls import path
from .views import main, about, contact, settings

urlpatterns = [
    path('', main, name='main'),
    path('about-the-team', about, name='about'),
    path('contact-us', contact, name='contact'),

    # Move this logged in apps into different location later meantime here.
    path('settings', settings, name='settings'),
]