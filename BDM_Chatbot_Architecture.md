# BDM Chatbot Application Architecture

## Overview

The BDM Chatbot application is a web-based chatbot that interacts with users, processes their queries, and provides responses. The application is built using Flask for the backend, SQLite for the database, and HTML/CSS/JavaScript for the frontend. It also integrates with external libraries and APIs for document processing and conversational AI.

## Components

### 1. Frontend

- **HTML/CSS/JavaScript**: The frontend is responsible for rendering the user interface and handling user interactions. It includes an input field for user messages and a chat box to display the conversation.
- **Libraries**:
  - `marked.js` and `DOMPurify.js` are used for rendering and sanitizing markdown content.
  - `Font Awesome` for icons.

### 2. Backend

- **Flask**: The Flask framework is used to create the web server and handle HTTP requests.
- **Routes**:
  - `/`: Serves the main HTML page.
  - `/chat`: Handles chat messages sent by the user and returns responses.

### 3. Database

- **SQLite**: The database stores chat history, including user questions, bot answers, and timestamps.
- **Tables**:
  - `chat_history`: Stores the chat logs.

### 4. Document Processing

- **PDF Reading**: The application reads and processes PDF documents using `PyPDFLoader`.
- **Text Splitting**: Documents are split into smaller chunks using `RecursiveCharacterTextSplitter`.
- **Vector Store**: The processed documents are stored in a vector store using `FAISS` for efficient retrieval.

### 5. Conversational AI

- **LangChain**: The application uses LangChain for building conversational AI chains.
- **Groq API**: Integrates with the Groq API for advanced conversational capabilities.

## Architecture Diagram

```plaintext
+---------------------+       +---------------------+       +---------------------+
|      Frontend       |       |       Backend       |       |      Database       |
|---------------------|       |---------------------|       |---------------------|
| - HTML/CSS/JS       | <---> | - Flask             | <---> | - SQLite            |
| - User Interface    |       | - Routes            |       | - chat_history      |
| - Chat Box          |       | - Chat Handling     |       | - Stores chat logs  |
+---------------------+       +---------------------+       +---------------------+

+---------------------+       +---------------------+       +---------------------+
|  Document Processing|       | Conversational AI   |       | External Libraries  |
|---------------------|       |---------------------|       |---------------------|
| - PyPDFLoader       |       | - LangChain         |       | - marked.js         |
| - Text Splitter     |       | - Groq API          |       | - DOMPurify.js      |
| - FAISS Vector Store|       | - ChatGroq          |       | - Font Awesome      |
+---------------------+       +---------------------+       +---------------------+
```
