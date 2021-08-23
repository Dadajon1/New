from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.urls import reverse_lazy
from django.views.generic import ListView,DetailView
from django.views.generic.edit import CreateView, DeleteView, UpdateView
from .models import Article
from django.shortcuts import  render
from django.core.files.storage import FileSystemStorage


class Articleview(ListView):
    model = Article
    template_name = 'article_list.html'

class Articledetailview(DetailView):
    model = Article
    template_name = 'article_detail.html'

class Articleupdateview(LoginRequiredMixin, UserPassesTestMixin, UpdateView):
    model = Article
    template_name = 'article_edit.html'
    fields = ('title','summary','body','photo')

    def test_func(self):
        obj = self.get_object()
        return obj.author == self.request.user


class Articledeleteview(LoginRequiredMixin, UserPassesTestMixin,DeleteView):
    model = Article
    template_name = 'article_delete.html'
    success_url = reverse_lazy('article_list')


    def test_func(self):
        obj = self.get_object()
        return obj.author == self.request.user


class ArticleCreateView(LoginRequiredMixin,UserPassesTestMixin, CreateView):
    model = Article
    template_name = 'article_new.html'
    fields = ('title','summary','body','photo','photo2','file')

    def form_valid(self, form):
        form.instance.author = self.request.user
        return super().form_valid(form)


    def test_func(self):
        return self.request.user.is_superuser