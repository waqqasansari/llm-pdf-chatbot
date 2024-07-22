from rest_framework import serializers

class ChatInputSerializer(serializers.Serializer):
    user_input = serializers.CharField()

class ChatOutputSerializer(serializers.Serializer):
    ai_response = serializers.CharField()


from .models import PDFDocument

class PDFDocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = PDFDocument
        fields = ('id', 'pdf_file', 'uploaded_at')