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
import uuid

app = FastAPI()



# Add this before defining your routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend URL, e.g. ["https://192.168.0.189:8080"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Initialize DB connection (adjust credentials as needed)
db = MariaDBConnection()
db.connect()

@app.post("/search-face/")
async def search_face(file: UploadFile = File(...), threshold: float = 0.65):
    # Save uploaded file to a temporary location in the current project folder
    tmp_filename = f"tmp_{uuid.uuid4().hex}.jpg"
    tmp_path = os.path.join(os.getcwd(), tmp_filename)
    try:
        contents = await file.read()
        with open(tmp_path, "wb") as tmp:
            tmp.write(contents)
        print(f"Temporary file saved at: {tmp_path}")
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
    finally:
        # Delete the temporary file after use
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
 # uvicorn api:app --host 0.0.0.0 --port 8000 --ssl-keyfile ./certs/key.pem --ssl-certfile ./certs/cert.pem --reload