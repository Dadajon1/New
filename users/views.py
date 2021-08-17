from django.shortcuts import render
from django.views.generic import CreateView
from .forms import Customusercreation
from django.urls import reverse_lazy

# Create your views here.

class Signupview(CreateView):
    form_class = Customusercreation
    success_url = reverse_lazy('login')
    template_name = 'registration/signup.html'

