# Academic Pal - Streamlit-based Application

**Academic Pal** is a Streamlit-based application designed for researchers and academics to streamline workflows related to academic paper management, experimental data extraction, and summarization. The application provides an intuitive interface for downloading, uploading, and processing research papers, powered by advanced embeddings and document retrieval methods.

---

## Features

### 1. User Authentication
- **Sign Up & Login:** Enables user authentication using Firebase for secure account management.
- **Firebase Integration:**
  - Firebase API for authentication.
  - Credential-based validation using Google Cloud.

### 2. Academic Paper Management
#### Paper Search & Download
- Fetch and download academic papers using the **Semantic Scholar API**.
- Support for storing downloaded papers in Google Cloud Storage (GCS).

#### Upload Papers
- Upload academic papers directly to GCS for later use.

### 3. Experimental Data Extraction
- Extract experimental details such as:
  - **Population size and characteristics**
  - **Study design**
  - **Research objectives**
  - **Effect sizes and p-values**
- Advanced processing of PDFs using **PyMuPDF** and embeddings from **HuggingFace BGE**.

### 4. Paper Summarization
- Summarizes research papers into concise, topic-based summaries.
- Includes further reading links for extended exploration.

### 5. Conversational Chatbot
- Academic assistant chatbot powered by:
  - **Groq Model (Llama3-70B)** for conversational intelligence.
  - Memory-based session management using LangChain’s `ConversationSummaryMemory`.
- Handles complex queries based on paper contents stored in GCS.

### 6. Vector Search & Retrieval
- Leverages **FAISS** for vector-based retrieval of document content.
- Provides a robust pipeline for question-answering over academic documents.

---

## Technologies Used

### Frontend
- [Streamlit](https://streamlit.io): For creating a highly interactive web-based interface.

### Backend
- **Semantic Scholar API:** For fetching research papers.
- **Firebase Authentication:** User authentication and session management.
- **Google Cloud Storage (GCS):** Cloud storage for academic papers and extracted details.

### Machine Learning
- **HuggingFace BGE Embeddings:** For document vectorization.
- **Groq AI:** For question-answering and summarization.
- **LangChain:**
  - Chains for Conversational Retrieval.
  - Document loaders and splitters.
  - Prompt templates for NLP workflows.

### Python Libraries
- `PyMuPDF`: For PDF parsing and loading.
- `FAISS`: For fast vector retrieval.
- `dotenv`: Environment variable management.
- `requests`: API communication with Firebase and Semantic Scholar.
- `firebase_admin`: Firebase backend SDK for authentication.
- `langchain_*`: Specialized tools for building chains and retrieval pipelines.

---

## Setup Instructions

### 1. Prerequisites
- **Python 3.8 or later**.
- **Docker** (optional for containerized deployment).
- Firebase project credentials for authentication.
- Google Cloud project setup with GCS.

### 2. Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/malshanCSe/AcademicPal.git
   cd AcademicPal
