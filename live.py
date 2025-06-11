import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis

# Initialize face analysis with GPU
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0)

# Load all embeddings from gdc profile folder
def load_gdc_embeddings(profile_folder):
    embeddings = []
    names = []
    for fname in os.listdir(profile_folder):
        if not (fname.lower().endswith('.jpg') or fname.lower().endswith('.png')):
            continue
        img_path = os.path.join(profile_folder, fname)
        img = cv2.imread(img_path)
        faces = app.get(img)
        if faces:
            embeddings.append(faces[0].embedding)
            names.append(os.path.splitext(fname)[0])
    return embeddings, names

def recognize_face(face_embedding, gdc_embeddings, gdc_names, threshold=0.35):
    for emb, name in zip(gdc_embeddings, gdc_names):
        sim = np.dot(face_embedding, emb) / (np.linalg.norm(face_embedding) * np.linalg.norm(emb))
        if sim > threshold:
            return name, sim
    return None, None

def main():
    gdc_folder = "gdc-profile"
    gdc_embeddings, gdc_names = load_gdc_embeddings(gdc_folder)
    print(f"Loaded {len(gdc_names)} profiles.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)
        for face in faces:
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            emb = face.embedding
            name, sim = recognize_face(emb, gdc_embeddings, gdc_names)
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if name:
                label = f"{name} ({sim:.2f})"
            else:
                label = "Unknown"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(frame, (x1, y1 - label_height - baseline), (x1 + label_width, y1), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, label, (x1, y1 - baseline), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        cv2.imshow("Live Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
