from django.urls import path
from .views import main, about, contact, register, login, files, settings, logout

urlpatterns = [
    path('', main, name='main'),
    path('about-the-team', about, name='about'),
    path('contact-us', contact, name='contact'),

    # Move this bottom two into authenticator app
    path('register', register, name='register'),
    path('login', login, name='login'),

    # MOve this logged in apps into different location later meantime here.
    path('files', files, name='files'),
    path('settings', settings, name='settings'),
    path('logout', logout, name='logout'),
]