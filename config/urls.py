
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('ckeditor/',include('ckeditor_uploader.urls')),
    path('admin/', admin.site.urls),
    path('users/',include('users.urls')),
    path('users/',include('django.contrib.auth.urls')),
    path('',include('pages.urls')),
    path('articles/', include('articles.urls')),
]


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)