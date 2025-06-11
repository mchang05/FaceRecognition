import cv2
import time
import requests
import numpy as np

class WebcamFaceSearcher:
    def __init__(self, api_url="http://127.0.0.1:8000/search-face/", interval_ms=100):
        self.api_url = api_url
        self.interval = interval_ms / 1000.0  # convert ms to seconds
        self.cap = cv2.VideoCapture(0)

    def capture_and_search(self):
        if not self.cap.isOpened():
            print("Cannot open webcam")
            return

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Encode frame as JPEG
                _, img_encoded = cv2.imencode('.jpg', frame)
                files = {'file': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}

                try:
                    response = requests.post(self.api_url, files=files)
                    if response.status_code == 200:
                        matches = response.json().get("matches", [])
                        print("Matches:", matches)
                    else:
                        print("API error:", response.text)
                except Exception as e:
                    print("Error calling API:", e)

                # Show the frame
                cv2.imshow('Webcam', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                time.sleep(self.interval)
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    searcher = WebcamFaceSearcher()
    searcher.capture_and_search()