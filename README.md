# 🚀 CODEDIVA Assignment - RAG Pipeline API

This project is a Retrieval-Augmented Generation (RAG) API built with **FastAPI**, **LangChain**, and **Google Gemini API**. It fulfills the assignment requirements by exposing endpoints to upload a document (`.pdf` or `.txt`) and answer queries based on its context, fully returning the answer along with cited sources.

## 🛠️ Technologies Used
- **API Framework**: FastAPI
- **RAG Orchestration**: LangChain
- **LLM Engine**: Google Gemini (`gemini-1.5-flash`)
- **Embeddings**: Google Gemini (`text-embedding-004`)
- **Vector Database**: FAISS (In-memory, local CPU)

## 📋 Prerequisites
- **Python**: 3.9 or higher
- **API Key**: A Google Gemini API Key (Get it free from [Google AI Studio](https://aistudio.google.com/))

---

## ⚙️ Setup & Installation

Run the following commands in your terminal to set up and start the API:

```bash
# 1. Navigate to the project folder
cd codediva-rag-assignment

# 2. Create and activate a virtual environment
py -m venv venv
source: # On macOS / Linux: venv/bin/activate        # On Windows use: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Google API Key
export GOOGLE_API_KEY="your-gemini-api-key-here"  # On Windows use: set GOOGLE_API_KEY="your-gemini-api-key-here"

# 5. Run the server
uvicorn main:app --reload