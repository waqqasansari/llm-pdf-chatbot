from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import ChatInputSerializer, ChatOutputSerializer
from .chatbot_logic import ChatbotLogic
import uuid

from rest_framework import viewsets
from rest_framework.parsers import MultiPartParser, FormParser
from .models import PDFDocument
from .serializers import PDFDocumentSerializer

class PDFUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    print('inside pdf')

    def post(self, request, *args, **kwargs):
        pdf_file = request.data.get('pdf_file')
        print('inside pdf post')
        print(pdf_file)

        # Check if the file is a PDF
        if not pdf_file.name.endswith('.pdf'):
            return Response({"error": "File must be a PDF."}, status=status.HTTP_400_BAD_REQUEST)

        # Create a new PDFDocument instance
        pdf_document = PDFDocument(pdf_file=pdf_file)
        pdf_document.save()

        # Serialize the saved document
        serializer = PDFDocumentSerializer(pdf_document)

        return Response(serializer.data, status=status.HTTP_201_CREATED)

    def get(self, request, *args, **kwargs):
        pdfs = PDFDocument.objects.all()
        serializer = PDFDocumentSerializer(pdfs, many=True)
        return Response(serializer.data)


class ChatbotView(APIView):
    chatbot_logic = None

    @classmethod
    def get_chatbot_logic(cls):
        if cls.chatbot_logic is None:
            print("Initializing ChatbotLogic...")
            cls.chatbot_logic = ChatbotLogic()
        return cls.chatbot_logic

    def post(self, request):
        serializer = ChatInputSerializer(data=request.data)
        if serializer.is_valid():
            user_input = serializer.validated_data['user_input']
            
            # Get or create a session ID
            session_id = request.session.get('chatbot_session_id')
            if not session_id:
                session_id = str(uuid.uuid4())
                request.session['chatbot_session_id'] = session_id

            # Use the shared ChatbotLogic instance
            chatbot = self.get_chatbot_logic()
            
            # Process the message
            response = chatbot.process_message(session_id, user_input)
            print(response)

            output_serializer = ChatOutputSerializer({'ai_response': response['answer']})
            return Response(output_serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)