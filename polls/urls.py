from django.urls import path

from . import views

urlpatterns = [
    # path('', views.index, name='index'),
    path(r'', views.UploadImg.as_view()),
    path(r'process', views.index, name='process'),

]