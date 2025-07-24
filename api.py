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

def find_profile_in_db_threaded(img_path, db, threshold=0.6):
    """Thread-safe version of find_profile_in_db"""
    face_app = get_face_app()
    
    img = cv2.imread(img_path)
    faces = face_app.get(img)
    if not faces:
        print("No face detected in query image.")
        return [], []

    largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    print(f"Largest face bbox: {largest_face.bbox.tolist()}")
    query_emb = largest_face.embedding
    matches = []

    # Use MariaDB vector search
    db.execute(
        """SELECT id, name, embedding, bbox, VEC_DISTANCE_COSINE(embedding, VEC_FromText(?)) as distance 
           FROM users 
           WHERE VEC_DISTANCE_COSINE(embedding, VEC_FromText(?)) < ? 
           ORDER BY distance""",
        (str(query_emb.tolist()), str(query_emb.tolist()), threshold)
    )
    for row in db.cursor:
        print(row['name'], row['distance'])
        matches.append(row)

    if not matches:
        print("No similar faces found in database.")
        fake_row = {
            'id': None,
            'name': 'unknown',
            'embedding': None,
            'bbox': str(largest_face.bbox.tolist()),
            'distance': 0
        }
        matches.append(fake_row)
    return matches, str(largest_face.bbox.tolist())

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

@app.post("/search-face/")
async def search_face(file: UploadFile = File(...), threshold: float = 0.65, db = Depends(get_db)):
    # Generate unique filename
    file_extension = os.path.splitext(file.filename)[1] or ".jpg"
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    # Save file
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)
    
    try:
        # Run face processing in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        matches, bbox = await loop.run_in_executor(
            executor, 
            find_profile_in_db_threaded, 
            file_path, 
            db, 
            threshold
        )
        
        results = [
            {
                "name": row['name'],
                "distance": float(row['distance']),
                "bbox": bbox
            }
            for row in matches
        ]
        return JSONResponse(content={"matches": results})
    except Exception as e:
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
    uvicorn.run(app, host="0.0.0.0", port=8000)