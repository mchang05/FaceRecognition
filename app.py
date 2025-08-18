from time import time
from insightface.app import FaceAnalysis
import numpy as np
import cv2
import os
from db import MariaDBConnection
import time

# Remove this global instance - it causes threading issues
# app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
# app.prepare(ctx_id=0)

def get_face_app():
    """Get a face analysis instance - each call creates a new one if needed"""
    try:
        app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0)
        return app
    except:
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=-1)
        return app

# Update all your functions to accept app parameter or create instance locally
def is_same_person(img1_path, img2_path, threshold=0.6):
    app = get_face_app()
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    faces1 = app.get(img1)
    faces2 = app.get(img2)

    if not faces1 or not faces2:
        print("Face not detected in one or both images.")
        return False

    emb1 = faces1[0].embedding

    for idx, face2 in enumerate(faces2):
        emb2 = face2.embedding
        sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        print(f"Comparing with face2_{idx}: Cosine similarity = {sim:.3f}")
        if sim > threshold:
            x1, y1, x2, y2 = [int(v) for v in face2.bbox]
            matched_face = img2[y1:y2, x1:x2]
            cv2.imwrite(f"matched_face_{idx}.jpg", matched_face)
            print(f"Match found and saved as matched_face_{idx}.jpg")
            return True

    print("No match found.")
    return False


def compare_and_draw(s_path, conference_folder, output_folder, threshold=0.35):
    app = get_face_app()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    img1 = cv2.imread(s_path)
    faces1 = app.get(img1)
    if not faces1:
        print(f"No face detected in {s_path}")
        return

    emb1 = faces1[0].embedding

    for fname in os.listdir(conference_folder):
        if not (fname.lower().endswith('.jpg') or fname.lower().endswith('.png')):
            continue
        img2_path = os.path.join(conference_folder, fname)
        img2 = cv2.imread(img2_path)
        faces2 = app.get(img2)
        found = False

        for idx, face2 in enumerate(faces2):
            emb2 = face2.embedding
            sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            if sim > threshold:
                x1, y1, x2, y2 = [int(v) for v in face2.bbox]
                cv2.rectangle(img2, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{sim:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 3
                thickness = 2
                (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                cv2.rectangle(img2, (x1, y1 - label_height - baseline), (x1 + label_width, y1), (0, 255, 0), cv2.FILLED)
                cv2.putText(img2, label, (x1, y1 - baseline), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
                out_path = os.path.join(output_folder, f"match_{fname}")
                cv2.imwrite(out_path, img2)
                print(f"Match found in {fname}, saved as {out_path}")
                found = True
                break

        if not found:
            print(f"No match found in {fname}")

def store_conference_embeddings(conference_folder, db):
    app = get_face_app()
    """
    Process all images in conference folder and store face embeddings in database
    """
    for fname in os.listdir(conference_folder):
        if not (fname.lower().endswith('.jpg') or fname.lower().endswith('.png')):
            continue

        img_path = os.path.join(conference_folder, fname)
        img = cv2.imread(img_path)
        faces = app.get(img)

        if not faces:
            print(f"No face detected in {fname}")
            continue

        for idx, face in enumerate(faces):
            try:
                # For MariaDB VECTOR type, pass as list of floats
                query = "INSERT INTO gallery (filename, embedding, bbox) VALUES (?, VEC_FromText(?), ?)"
                db.execute(query, (fname, str(face.embedding.tolist()), str(face.bbox.tolist())))
                db.commit()
                print(f"Stored embedding for face {idx} from {fname}")
            except Exception as e:
                print(f"Error storing {fname}: {str(e)}")

def store_gdc_profile_embeddings(profile_folder, db):
    """
    Process all images in the profile folder and store the largest face embedding in the database.
    Only the largest face (by area) in each image is stored.
    Measures and prints timing for each step.
    """
    app = get_face_app()
    
    # Overall timing
    total_start_time = time.time()
    total_files = 0
    total_faces_detected = 0
    
    print(f"🚀 Starting profile embedding conversion for folder: {profile_folder}")
    print("=" * 80)
    
    for fname in os.listdir(profile_folder):
        if not (fname.lower().endswith('.jpg') or fname.lower().endswith('.png')):
            continue

        total_files += 1
        file_start_time = time.time()
        
        # Step 1: Load image
        load_start = time.time()
        img_path = os.path.join(profile_folder, fname)
        img = cv2.imread(img_path)
        load_time = time.time() - load_start
        
        # Step 2: Face detection and embedding
        detection_start = time.time()
        faces = app.get(img)
        detection_time = time.time() - detection_start

        if not faces:
            file_time = time.time() - file_start_time
            print(f"❌ {fname:<25} | No face detected | Total: {file_time:.3f}s")
            continue

        total_faces_detected += 1
        
        # Step 3: Find largest face
        processing_start = time.time()
        largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        face_area = (largest_face.bbox[2] - largest_face.bbox[0]) * (largest_face.bbox[3] - largest_face.bbox[1])
        processing_time = time.time() - processing_start

        # Step 4: Database storage
        db_start = time.time()
        try:
            query = "INSERT INTO users (name, embedding, bbox) VALUES (?, VEC_FromText(?), ?)"
            db.execute(query, (fname, str(largest_face.embedding.tolist()), str(largest_face.bbox.tolist())))
            db.commit()
            db_time = time.time() - db_start
            
            # Total time for this file
            file_time = time.time() - file_start_time
            
            print(f"✅ {fname:<25} | "
                  f"Load: {load_time:.3f}s | "
                  f"Detect: {detection_time:.3f}s | "
                  f"Process: {processing_time:.3f}s | "
                  f"DB: {db_time:.3f}s | "
                  f"Total: {file_time:.3f}s | "
                  f"Area: {face_area:.0f}px")
            
        except Exception as e:
            db_time = time.time() - db_start
            file_time = time.time() - file_start_time
            print(f"❌ {fname:<25} | Error storing: {str(e)[:50]}... | Time: {file_time:.3f}s")

    # Summary statistics
    total_time = time.time() - total_start_time
    
    print("=" * 80)
    print(f"📊 PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"Total files processed: {total_files}")
    print(f"Faces successfully detected: {total_faces_detected}")
    print(f"Success rate: {(total_faces_detected/total_files*100):.1f}%" if total_files > 0 else "No files processed")
    print(f"Total processing time: {total_time:.3f}s")
    print(f"Average time per file: {(total_time/total_files):.3f}s" if total_files > 0 else "N/A")
    print(f"Average time per successful detection: {(total_time/total_faces_detected):.3f}s" if total_faces_detected > 0 else "N/A")
    print(f"Processing speed: {(total_files/total_time):.2f} files/second" if total_time > 0 else "N/A")
    
    return {
        "total_files": total_files,
        "successful_detections": total_faces_detected,
        "total_time": total_time,
        "avg_time_per_file": total_time/total_files if total_files > 0 else 0,
        "files_per_second": total_files/total_time if total_time > 0 else 0
    }

def find_similar_in_db(img_path, db, threshold=0.6):
    app = get_face_app()
    """
    Compare the embedding of img_path to all embeddings in the database.
    Prints matches above the threshold.
    """
    img = cv2.imread(img_path)
    faces = app.get(img)
    if not faces:
        print("No face detected in query image.")
        return []

    query_emb = faces[0].embedding
    matches = []

    # Use MariaDB vector search (requires MariaDB 11+ and VECTOR type)
    db.execute(
        """SELECT id, filename, embedding, bbox, VEC_DISTANCE_COSINE(embedding, VEC_FromText(?)) as distance 
           FROM gallery 
           WHERE VEC_DISTANCE_COSINE(embedding, VEC_FromText(?)) < ? 
           ORDER BY distance""",
        (str(query_emb.tolist()), str(query_emb.tolist()), threshold)
    )
    for row in db.cursor:
        print(row['filename'], row['distance'])
        matches.append(row)

    if not matches:
        print("No similar faces found in database.")
    return matches

def find_profile_in_db(img_path, db, threshold=0.6):
    app = get_face_app()
    """
    Compare the embedding of img_path to all embeddings in the database.
    Prints matches above the threshold.
    """
    img = cv2.imread(img_path)
    faces = app.get(img)
    if not faces:
        print("No face detected in query image.")
        return [], []

    largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    print(f"Largest face bbox: {largest_face.bbox.tolist()}")
    query_emb = largest_face.embedding
    matches = []

    # Use MariaDB vector search (requires MariaDB 11+ and VECTOR type)
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

def get_embbeding(img_path):
    app = get_face_app()
    
    img = cv2.imread(img_path)
    faces = app.get(img)
    if not faces:
        print("No face detected in query image.")
        return None

    largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return largest_face.embedding.tolist()  # Return as a list of floats for compatibility with MariaDB VECTOR type

# Example usage:
if __name__ == "__main__":
    db = MariaDBConnection()
    db.connect()
    find_profile_in_db('test.jpg', db, threshold=0.65)
    # store_conference_embeddings('conference', db)
    # store_gdc_profile_embeddings('test', db)
    db.close()
    
    # result = is_same_person('gdc-profile/AlisaSimaroj.jpg', 'conference/20250219_AIA01752.jpg')
    # print("Same person." if result else "Different persons.")
    # compare_and_draw('gdc-profile/CheeSoongLeong.jpg', 'conference', 'output')#conference, dinner