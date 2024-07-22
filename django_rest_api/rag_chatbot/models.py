# Create your models here.
from django.db import models

class ChatSession(models.Model):
    session_id = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Chat Session {self.session_id}"


class PDFDocument(models.Model):
    pdf_file = models.FileField(upload_to='pdfs/')
    uploaded_at = models.DateTimeField(auto_now_add=True)