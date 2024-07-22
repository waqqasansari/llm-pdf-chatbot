from django.contrib import admin
from .views import PDFUploadView, ChatbotView
from django.urls import path


urlpatterns = [
    path('upload/', PDFUploadView.as_view()),
    path('chat/', ChatbotView.as_view(), name='chatbot'),
]