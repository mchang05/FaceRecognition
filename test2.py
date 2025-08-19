import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

img_path = 'test/AlvinHo.jpg'
img = cv2.imread(img_path)

# Add 50 pixels padding to all sides
padding = 50
padded_img = cv2.copyMakeBorder(
    img, 
    padding, padding, padding, padding,  # top, bottom, left, right
    cv2.BORDER_CONSTANT, 
    value=[0, 0, 0]  # Black padding (BGR format)
)

# Use the padded image for face detection
faces = app.get(padded_img)
print(f"Original image shape: {img.shape}")
print(f"Padded image shape: {padded_img.shape}")
print(f"Number of faces detected: {len(faces)}")
print(faces)

# Optional: Save the padded image to see the result
cv2.imwrite('test/AlvinHo_padded.jpg', padded_img)
print("Padded image saved as 'test/AlvinHo_padded.jpg'")