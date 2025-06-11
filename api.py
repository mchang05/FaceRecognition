from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import tempfile
from db import MariaDBConnection
from app import app as face_app, find_similar_in_db, find_profile_in_db
from dotenv import load_dotenv
import os 

app = FastAPI()

load_dotenv()
db_password = os.getenv("DB_PASSWORD")
db_user = os.getenv("DB_USER")
db_name = os.getenv("DB_NAME")

# Add this before defining your routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend URL, e.g. ["https://192.168.0.189:8080"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print(db_password, db_user)
# Initialize DB connection (adjust credentials as needed)
db = MariaDBConnection(user=db_user, password=db_password, database=db_name)
db.connect()

@app.post("/search-face/")
async def search_face(file: UploadFile = File(...), threshold: float = 0.65):
    # Save uploaded file to a temporary location
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process uploaded file: {str(e)}")

    # Use find_similar_in_db to search for similar faces
    try:
        matches, bbox = find_profile_in_db(tmp_path, db, threshold=threshold)
        # Format results for JSON response
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
    
    
    # uvicorn api:app --host 0.0.0.0 --port 8000 --ssl-keyfile ./certs/key.pem --ssl-certfile ./certs/cert.pem --reload