from django.db import models

class File(models.Model):
    Img = models.ImageField(upload_to='media/')