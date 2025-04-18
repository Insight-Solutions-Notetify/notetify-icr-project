from django.urls import path
from .views import main, about, contact, files, settings, logout

urlpatterns = [
    path('', main, name='main'),
    path('about-the-team', about, name='about'),
    path('contact-us', contact, name='contact'),

    # Move this logged in apps into different location later meantime here.
    path('files', files, name='files'),
    path('settings', settings, name='settings'),
    path('logout', logout, name='logout'),
]