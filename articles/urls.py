from django.urls import path
from .views import Articleview,Articledetailview, ArticleCreateView, Articledeleteview,Articleupdateview


urlpatterns = [
    path('<int:pk>/edit/', Articleupdateview.as_view(), name="article_edit"),
    path('<int:pk>/', Articledetailview.as_view(), name='article_detail'),
    path('<int:pk>/delete/', Articledeleteview.as_view(), name='article_delete'),
    path('new/', ArticleCreateView.as_view(), name='article_new'),
    path('', Articleview.as_view(), name='article_list'),
]