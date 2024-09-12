# **LLM-PDF-Chatbot**

## **Overview**

This repository contains a Django REST Framework project that enables users to upload PDF files and interact with them through a chatbot. The chatbot utilizes the GPT-3.5 model, OpenAI embeddings, and Pinecode vector database to provide intelligent responses. Additionally, Langchain is used to create a chat QA chain that maintains conversation history.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.12
- pip (Python package manager)
- OpenAI API key
- Pinecode
- Langchain

## **Features**

- **PDF Upload**: Users can upload PDF files, which are then accessible for content-based queries.
- **Chatbot Integration**: The chatbot uses the GPT-3.5 model and OpenAI embeddings to generate responses to user queries about the uploaded PDF content.
- **Conversation History**: Langchain is used to create a chat QA chain that maintains a record of the conversation history.

## **API Endpoints**

### 1. Upload PDF File

- **Endpoint**: `/api/upload/`
- **Method**: `POST`
- **Body**: In form file name = `pdf_file`
- **Description**: Upload a PDF file for chatbot interaction.

### 2. Chat with PDF

- **Endpoint**: `/api/chat/`
- **Method**: `POST`
- **Description**: Interact with the uploaded PDF file through the
  chatbot.
- **Body**:

```json
{
  "user_input": "Your question or message here"
}
```

The API will respond with a JSON object containing the chatbot's response:

```json
{
  "ai_response": "The chatbot's response to your input"
}
```

## **Technology Stack**

- **Backend**: Django REST Framework
- **Language Model**: GPT-3.5
- **Embeddings**: OpenAI
- **Vector Database**: Pinecode
- **Chat QA Chain**: Langchain

## **Getting Started**

### Installation

1. Clone the repository: `git clone https://github.com/your-username/llm-pdf-chatbot.git`
2. Navigate to the project directory: `cd llm-pdf-chatbot/`
3. Create a virtual environment: `python3 -m venv venv`
4. Activate the virtual environment: `source venv/bin/activate`
5. Install dependencies: `pip install -r requirements.txt`
6. Navigate to the project directory: `cd django_rest_api/`
7. Run migrations: `python manage.py makemigrations`
8. Apply database migrations: `python manage.py migrate`
9. Run the development server: `python manage.py runserver`

### Usage

1. Access the application at `http://localhost:8000/`
2. Upload a PDF file using the `/api/upload/` endpoint.
3. Interact with the uploaded PDF file using the `/api/chat/` endpoint.

## **License**

This project is licensed under the MIT License. See the `LICENSE` file for details.
