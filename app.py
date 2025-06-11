from insightface.app import FaceAnalysis
import numpy as np
import cv2
import os
from db import MariaDBConnection

app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0)

def is_same_person(img1_path, img2_path, threshold=0.6):
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
    """
    for fname in os.listdir(profile_folder):
        if not (fname.lower().endswith('.jpg') or fname.lower().endswith('.png')):
            continue

        img_path = os.path.join(profile_folder, fname)
        img = cv2.imread(img_path)
        faces = app.get(img)

        if not faces:
            print(f"No face detected in {fname}")
            continue

        # Find the largest face by area
        largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        try:
            query = "INSERT INTO users (name, embedding, bbox) VALUES (?, VEC_FromText(?), ?)"
            db.execute(query, (fname, str(largest_face.embedding.tolist()), str(largest_face.bbox.tolist())))
            db.commit()
            print(f"Stored largest face embedding from {fname}")
        except Exception as e:
            print(f"Error storing {fname}: {str(e)}")

def find_similar_in_db(img_path, db, threshold=0.6):
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

# Example usage:
if __name__ == "__main__":
    db = MariaDBConnection()
    db.connect()
    # find_similar_in_db('gdc-profile/LinLingToh.jpg', db, threshold=0.65)
    # store_conference_embeddings('conference', db)
    store_gdc_profile_embeddings('gdc-profile', db)
    db.close()
    
    # result = is_same_person('gdc-profile/AlisaSimaroj.jpg', 'conference/20250219_AIA01752.jpg')
    # print("Same person." if result else "Different persons.")
    # compare_and_draw('gdc-profile/CheeSoongLeong.jpg', 'conference', 'output')#conference, dinner