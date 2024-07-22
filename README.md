Here is a sample README file for your GitHub repository:

**LLM-PDF-Chatbot**
=====================

**Overview**
-----------

This repository contains a Django REST Framework project that enables users to upload PDF files and interact with them through a chatbot. The chatbot utilizes the GPT-3.5 model, OpenAI embeddings, and Pinecode vector database to provide intelligent responses. Additionally, Langchain is used to create a chat QA chain that maintains conversation history.

**Features**
------------

- **PDF Upload**: Users can upload PDF files, which are then accessible for content-based queries.
- **Chatbot Integration**: The chatbot uses the GPT-3.5 model and OpenAI embeddings to generate responses to user queries about the uploaded PDF content.
- **Conversation History**: Langchain is used to create a chat QA chain that maintains a record of the conversation history.
- **Responsive UI**: The application features a responsive user interface, ensuring a seamless experience across various devices.

**API Endpoints**
-----------------

### 1. Upload PDF File

- **Endpoint**: `/api/upload-pdf/`
- **Method**: `POST`
- **Description**: Upload a PDF file for chatbot interaction.

### 2. Chat with PDF

- **Endpoint**: `/api/chat/`
- **Method**: `POST`
- **Description**: Interact with the uploaded PDF file through the chatbot.

**Technology Stack**
--------------------

- **Backend**: Django REST Framework
- **Language Model**: GPT-3.5
- **Embeddings**: OpenAI
- **Vector Database**: Pinecode
- **Chat QA Chain**: Langchain

**Getting Started**
---------------

### Installation

1. Clone the repository: `git clone https://github.com/your-username/llm-pdf-chatbot.git`
2. Navigate to the project directory: `cd llm-pdf-chatbot`
3. Create a virtual environment: `python3 -m venv venv`
4. Activate the virtual environment: `source venv/bin/activate`
5. Install dependencies: `pip install -r requirements.txt`
6. Apply database migrations: `python manage.py migrate`
7. Create a superuser: `python manage.py createsuperuser`
8. Run the development server: `python manage.py runserver`

### Usage

1. Access the application at `http://localhost:8000/`
2. Upload a PDF file using the `/api/upload-pdf/` endpoint.
3. Interact with the uploaded PDF file using the `/api/chat/` endpoint.

**License**
---------

This project is licensed under the MIT License. See the `LICENSE` file for details.

**Contributing**
------------

Contributions are welcome If you find any issues or want to add new features, feel free to open a pull request.
