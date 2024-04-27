from django.contrib import admin
from django.urls import path
from home import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [

    path("" , views.index, name='home'),
    path("about" , views.about, name='about'),
    path("services" , views.services, name='services'),
    path("contact" , views.contact, name='contact'),
    path("success" , views.success, name='success'),
    path("file" , views.file, name='file'),
    path("show" , views.show, name='show')
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)