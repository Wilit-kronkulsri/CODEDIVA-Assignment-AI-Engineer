import os
import uuid
import tempfile
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings


class GeminiEmbeddings(Embeddings):
    def __init__(self):
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = genai.embed_content(
            model="models/gemini-embedding-2",
            content=texts,
            task_type="retrieval_document"
        )
        return response['embedding']

    def embed_query(self, text: str) -> List[float]:
        response = genai.embed_content(
            model="models/gemini-embedding-2",
            content=text,
            task_type="retrieval_query"
        )
        return response['embedding']


app = FastAPI(
    title="CODEDIVA RAG API",
    description="Simple RAG Pipeline using FastAPI, FAISS, and Google Gemini SDK"
)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


DOC_STORES = {}


class QueryRequest(BaseModel):
    doc_id: str
    query: str


class SourceDetail(BaseModel):
    source_file: str
    page: Optional[int]
    content_snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDetail]


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    message: str



@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):

    if not os.environ.get("GOOGLE_API_KEY"):
        raise HTTPException(status_code=500, detail="Missing GOOGLE_API_KEY environment variable.")

    ext = file.filename.split('.')[-1].lower()
    if ext not in ["pdf", "txt"]:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use .pdf or .txt")

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name

    try:
        if ext == "pdf":
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path, encoding="utf-8")

        documents = loader.load()

        for doc in documents:
            doc.metadata["source"] = file.filename

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        embeddings = GeminiEmbeddings()
        vector_store = FAISS.from_documents(documents=splits, embedding=embeddings)

        doc_id = str(uuid.uuid4())
        DOC_STORES[doc_id] = vector_store

        return UploadResponse(
            doc_id=doc_id,
            filename=file.filename,
            message="Document indexed successfully."
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):

    if not os.environ.get("GOOGLE_API_KEY"):
        raise HTTPException(status_code=500, detail="Missing GOOGLE_API_KEY environment variable.")

    if request.doc_id not in DOC_STORES:
        raise HTTPException(status_code=404, detail="Document ID not found. Please upload a document first.")

    vector_store = DOC_STORES[request.doc_id]
    docs = vector_store.similarity_search(request.query, k=3)

    context_text = "\n\n".join([doc.page_content for doc in docs])

    system_prompt = (
        "You are a helpful assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Always base your answer strictly on the provided context."
    )

    try:
        model = genai.GenerativeModel(
            model_name="gemini-3.5-flash",
            system_instruction=system_prompt
        )

        full_prompt = f"Context:\n{context_text}\n\nQuestion: {request.query}"
        response = model.generate_content(full_prompt)
        answer = response.text

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini Generation Error: {str(e)}")

    sources = []
    for doc in docs:
        sources.append(SourceDetail(
            source_file=doc.metadata.get("source", "Unknown"),
            page=doc.metadata.get("page"),
            content_snippet=doc.page_content[:150] + "..."
        ))

    return QueryResponse(answer=answer, sources=sources)