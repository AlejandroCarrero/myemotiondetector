import cv2
from deepface import DeepFace

# Initialize webcam
cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Detect faces and analyze emotions
        results = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False
        )

        # Ensure results is a list (handles multiple faces)
        if not isinstance(results, list):
            results = [results]

        for result in results:
            face = result['region']

            # Skip invalid detections
            if face['w'] <= 0 or face['h'] <= 0:
                continue

            x, y, w, h = face['x'], face['y'], face['w'], face['h']
            dominant_emotion = result['dominant_emotion']

            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 128, 255), 2)

            # Show the emotion label
            cv2.putText(frame, dominant_emotion,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, cv2.LINE_AA)

    except Exception:
        cv2.putText(frame, "Scanning for faces...", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("Emotion Detector", frame)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
