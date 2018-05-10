from django.urls import path

from . import views

urlpatterns = [
    # path('', views.index, name='index'),
    path(r'', views.UploadImg.as_view()),
    path(r'process', views.index, name='process'),
    path(r'up_2_img', views.upload_2_img.as_view()),
    path(r'distance', views.measure_distance, name='distance'),

]