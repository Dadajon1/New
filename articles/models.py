from django.contrib.auth import get_user_model
from django.db import models
from django.urls import reverse
from ckeditor.fields import RichTextField

class Article(models.Model):
    title = models.CharField(max_length=300)
    summary = models.CharField(max_length=300, blank=True)
    body = RichTextField()
    photo = models.ImageField(upload_to='images/', blank=True)
    file = models.FileField(upload_to='files/', blank=True)
    date = models.DateTimeField(auto_now_add=True)
    author = models.ForeignKey(
        get_user_model(),
        on_delete= models.CASCADE,
    )
    def __str__(self):
        return self.title

    def get_absolute_url(self):
        return reverse('article_detail', args=[str(self.id)])