import os
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import database
import rag
import agent

# Lifecycle management for indexing files on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure folder structure exists
    os.makedirs("data/knowledge_base", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    
    # Initialize RAG vector index
    try:
        rag.build_or_load_index()
    except Exception as e:
        print(f"Startup Warning: Could not index files: {e}")
    yield

app = FastAPI(title="Customer Support AI Agent Developer Dashboard", lifespan=lifespan)

# Define models
class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []

# API Endpoints
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        result = agent.run_agent(request.message, request.history)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/db/orders")
async def get_orders():
    try:
        return database.list_orders()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/db/tickets")
async def get_tickets():
    try:
        return database.list_tickets()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    dest_path = os.path.join("data", "knowledge_base", file.filename)
    try:
        # Save uploaded PDF file to knowledge base folder
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Re-build RAG index with the new file
        rag.build_or_load_index()
        
        return {"status": "success", "message": f"Successfully uploaded and indexed '{file.filename}'."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

# Serve frontend single-page dashboard
@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = "static/index.html"
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="""
        <html>
            <head><title>Setup Required</title></head>
            <body style="font-family: sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; background: #121214; color: #fff;">
                <h1>Backend is Running!</h1>
                <p>The static files for the dashboard are not created yet. Please implement the frontend static files.</p>
            </body>
        </html>
    """)

# Mount static folder for assets (css, js, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    # Read port from environment or default to 8000
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
