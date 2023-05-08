from django.urls import path
from . import views


# route_paths for the database app

urlpatterns = [
    path('classify/', views.classify),
]
