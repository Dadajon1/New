from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from .forms import Customusercreation, Customuserchange
from .models import CustomUser

# Register your models here.
class Customuseradmin(UserAdmin):
    add_form = Customusercreation
    form = Customuserchange
    model= CustomUser
    list_display = ['email','age','username','is_staff','adress']


admin.site.register(CustomUser,Customuseradmin)