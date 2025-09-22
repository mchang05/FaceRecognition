from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import os
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from contextlib import asynccontextmanager
from db import MariaDBConnection
from datetime import datetime
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct
from qdrant import search_similar_faces
import time

# Thread-local storage for face analysis instances
thread_local = threading.local()

def get_face_app():
    """Get thread-local face analysis instance"""
    if not hasattr(thread_local, 'face_app'):
        try:
            from insightface.app import FaceAnalysis
            thread_local.face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            thread_local.face_app.prepare(ctx_id=0)
            print("Using CUDA for face analysis")
        except Exception as e:
            print(f"CUDA failed, falling back to CPU: {e}")
            from insightface.app import FaceAnalysis
            thread_local.face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            thread_local.face_app.prepare(ctx_id=-1)
            print("Using CPU for face analysis")
    return thread_local.face_app

# Thread pool for CPU-intensive face processing
executor = ThreadPoolExecutor(max_workers=4)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    """Dependency to get a fresh database connection for each request"""
    db = MariaDBConnection()
    try:
        db.connect()
        yield db
    finally:
        db.close()

# Create uploads directory if it doesn't exist
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def get_embedding_threaded(img_path):
    """Thread-safe version of get_embedding"""
    face_app = get_face_app()
    
    img = cv2.imread(img_path)
    faces = face_app.get(img)
    if not faces:
        print("No face detected in query image.")
        return None

    largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return largest_face.embedding.tolist()

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url="http://localhost:6333", 
    api_key="123456",  # Use your actual API key
    prefer_grpc=True
)

def search_similar_faces_threaded(img_path, limit=5, score_threshold=0.6):
    
    try:
        search_results = search_similar_faces("aia_lf_2025", img_path, limit=limit, score_threshold=score_threshold)
        print(search_results)
    
            
        return search_results
        
    except Exception as e:
        print(f"❌ Error searching faces in Qdrant: {e}")
        # # Return unknown result on error
        # fake_result = {
        #     'person_name': 'unknown',
        #     'similarity_score': 0.0,
        #     'bbox': bbox,
        #     'filename': 'unknown'
        # }
        # return [fake_result], bbox

            
@app.post("/search-face/")
async def search_face(file: UploadFile = File(...), threshold: float = 0.3, limit: int = 5):
    """
    Search for similar faces using Qdrant vector database
    
    Args:
        file: Image file to search
        threshold: Similarity threshold (0.0 to 1.0, higher = more similar)
        limit: Maximum number of results to return
    """
    start_time = time.time()
    
    # Generate unique filename
    file_extension = os.path.splitext(file.filename)[1] or ".jpg"
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    # Save file
    save_start = time.time()
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)
    save_time = time.time() - save_start
    
    try:
        # Run face processing in thread pool to avoid blocking
        search_start = time.time()
        loop = asyncio.get_event_loop()
        matches = await loop.run_in_executor(
            executor, 
            search_similar_faces_threaded, 
            file_path,
            limit,
            threshold
        )
        search_time = time.time() - search_start
        
        # Format results for API response
        results = []
        for match in matches:
            print(match)
            results.append({
                "name": match["name"],
                "distance": 1.0 - float(match["similarity_score"]),  # Convert similarity to distance
                "photo_path": match["photo_path"],
                "qr_code": match["qr_code"],
            })
        
        total_time = time.time() - start_time
        
        response_data = {
            "matches": results,
        }
        
        print(f"✅ Search completed: {len(results)} matches in {total_time:.3f}s")
        return JSONResponse(content=response_data)
        
    except Exception as e:
        error_time = time.time() - start_time
        print(f"❌ Search failed after {error_time:.3f}s: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during face search: {str(e)}")
    finally:
        # Clean up - remove file after processing
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/get_embed/")
async def get_embed(file: UploadFile = File(...), db = Depends(get_db)):
    # Generate unique filename
    file_extension = os.path.splitext(file.filename)[1] or ".jpg"
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    # Save file
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)
    
    try:
        # Run face processing in thread pool
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            executor, 
            get_embedding_threaded, 
            file_path
        )
        return JSONResponse(content={"embed": embedding})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during face processing: {str(e)}")
    finally:
        # Clean up - remove file after processing
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    
    # # SSL configuration
    ssl_keyfile = "certs/key.pem"
    ssl_certfile = "certs/cert.pem"
    
    # Check if SSL files exist
    if os.path.exists(ssl_keyfile) and os.path.exists(ssl_certfile):
        print("Starting server with HTTPS on port 8443...")
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8000,  # Standard HTTPS port alternative
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile
        )
    else:
        print("SSL certificates not found, starting with HTTP...")
        uvicorn.run(app, host="0.0.0.0", port=8000)