from django.contrib import admin
from django.urls import path, include
from .views import ChatbotView

urlpatterns = [
    path('ask/', ChatbotView.as_view(), name='chatbot-ask'),
]