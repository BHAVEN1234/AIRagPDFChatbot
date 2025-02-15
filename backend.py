import warnings
import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from PyPDF2 import PdfReader, errors
from pydantic import BaseModel

warnings.filterwarnings("ignore")

# API Keys and Configuration Storage (in-memory)
API_KEYS = {}
QDRANT_CONFIG = {}
MODEL_CONFIG = {}

# Embedding Model Options
EMBEDDING_MODEL_OPTIONS = {
    "all-MiniLM-L6-v2": "all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    # Add more options here
}

# Upload Directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF."""
    try:
        reader = PdfReader(pdf_path)
        text_chunks = []

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                chunks = text.split("\n\n")
                text_chunks.extend([chunk.strip() for chunk in chunks if chunk.strip()])
            else:
                print(f"Warning: No text extracted from page {page_num + 1}.")

        return text_chunks
    except errors.DependencyError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing error: {e}")


def store_pdf_embeddings(pdf_path, collection_name="pdf_embeddings", qdrant_api_key=None,
                         qdrant_cloud_url=None, embedding_model_name=None):
    """Extracts text, generates embeddings, and stores them in Qdrant."""
    text_chunks = extract_text_from_pdf(pdf_path)
    if not text_chunks:
        print("No valid text extracted. Skipping embedding storage.")
        return False

    qdrant_client = QdrantClient(url=qdrant_cloud_url, api_key=qdrant_api_key)  # Use API key and URL from arguments

    embedding_model = SentenceTransformer(embedding_model_name)  # Load model

    if qdrant_client.collection_exists(collection_name):
        qdrant_client.delete_collection(collection_name)

    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    for idx, chunk in enumerate(text_chunks):
        vector = embedding_model.encode(chunk).tolist()
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[PointStruct(id=idx, vector=vector, payload={"text": chunk})]
        )
    return True


def retrieve_similar_chunks(query, collection_name="pdf_embeddings", qdrant_api_key=None,
                             qdrant_cloud_url=None, embedding_model_name=None, top_k=3):
    """Retrieves top_k similar text chunks from Qdrant."""
    qdrant_client = QdrantClient(url=qdrant_cloud_url, api_key=qdrant_api_key)  # Use API key and URL from arguments

    embedding_model = SentenceTransformer(embedding_model_name)  # Load model

    query_vector = embedding_model.encode(query).tolist()
    search_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k
    )
    return [result.payload['text'] for result in search_results]


def generate_response(query, groq_api_key=None, qdrant_api_key=None, qdrant_cloud_url=None,
                      embedding_model_name=None):
    """Generates a response based on retrieved context."""
    context_chunks = retrieve_similar_chunks(query, qdrant_api_key=qdrant_api_key,
                                              qdrant_cloud_url=qdrant_cloud_url,
                                              embedding_model_name=embedding_model_name)

    if not context_chunks:
        return "No relevant context found in PDF."

    context = "\n".join(context_chunks)
    prompt = f"""Answer based on this context:

    Context:
    {context}

    Query: {query}
    Answer:
    """

    try:
        groq_llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")  # Use API key from argument
        response = groq_llm.predict(prompt)
        return response
    except Exception as e:
        return f"Error generating response: {e}"


app = FastAPI()


class Settings(BaseModel):
    qdrant_api_key: str
    groq_api_key: str
    qdrant_cloud_url: str
    embedding_model: str  # Add embedding model selection


@app.post("/set-settings")
async def set_settings(settings: Settings):
    """Sets the Qdrant API key, Groq API key, and Qdrant Cloud URL."""
    API_KEYS["qdrant_api_key"] = settings.qdrant_api_key
    API_KEYS["groq_api_key"] = settings.groq_api_key
    QDRANT_CONFIG["qdrant_cloud_url"] = settings.qdrant_cloud_url
    MODEL_CONFIG["embedding_model"] = settings.embedding_model
    return JSONResponse({"message": "Settings stored."})


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Handles PDF upload, stores text embeddings in Qdrant."""
    if "qdrant_api_key" not in API_KEYS or "qdrant_cloud_url" not in QDRANT_CONFIG or "embedding_model" not in MODEL_CONFIG:
        raise HTTPException(status_code=400, detail="Qdrant API key, Cloud URL, and Embedding Model must be set.")

    try:
        pdf_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        status = store_pdf_embeddings(pdf_path,
                                        qdrant_api_key=API_KEYS["qdrant_api_key"],
                                        qdrant_cloud_url=QDRANT_CONFIG["qdrant_cloud_url"],
                                        embedding_model_name=MODEL_CONFIG["embedding_model"])

        if status:
            return JSONResponse({"message": "PDF processed and embeddings stored."})
        else:
            return JSONResponse({"message": "No valid text found in PDF."}, status_code=400)
    except Exception as e:
         return JSONResponse({"message": f"upload PDF error: {e}"}, status_code=500)


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
async def query_pdf(query_request: QueryRequest):
    """Retrieves answers based on user queries using Groq LLM."""
    if "groq_api_key" not in API_KEYS or "qdrant_api_key" not in API_KEYS or "qdrant_cloud_url" not in QDRANT_CONFIG or "embedding_model" not in MODEL_CONFIG:
        raise HTTPException(status_code=400, detail="Groq API key, Qdrant API key, Cloud URL, and Embedding Model must be set.")

    query = query_request.query.strip()

    if not query:
        return JSONResponse({"answer": "Invalid query."}, status_code=400)

    answer = generate_response(query,
                                 groq_api_key=API_KEYS["groq_api_key"],
                                 qdrant_api_key=API_KEYS["qdrant_api_key"],
                                 qdrant_cloud_url=QDRANT_CONFIG["qdrant_cloud_url"],
                                 embedding_model_name=MODEL_CONFIG["embedding_model"])
    return JSONResponse({"answer": answer})
