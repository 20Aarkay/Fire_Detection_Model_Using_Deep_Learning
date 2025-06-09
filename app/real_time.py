import cv2
import numpy as np
from tensorflow.keras.preprocessing import image

def gen_frames(model, class_names):
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not start camera.")

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Resize frame to model input size (e.g., 224x224)
        img = cv2.resize(frame, (224, 224))

        # Preprocess for model
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        preds = model.predict(img_array)[0]
        predicted_index = np.argmax(preds)
        predicted_label = class_names[predicted_index]
        confidence = preds[predicted_index]

        # Optionally, draw prediction text on frame
        label_text = f"{predicted_label}: {confidence*100:.2f}%"
        color = (0, 0, 255) if predicted_label == 'fire' else (0, 255, 255) if predicted_label == 'smoke' else (0, 255, 0)
        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # If fire detected, you can trigger an alarm here
        if predicted_label == 'fire' and confidence > 0.7:
            # Example: Draw red rectangle or do other alert logic
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    camera.release()
