import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
import tensorflow as tf

# Load the trained model
model_path = r"C:\Users\gouri\Downloads\sign_language_model_sequential1.h5"
model = tf.keras.models.load_model(model_path)

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

def preprocess_frame(frame):
    # Resize frame to match the input size of the model
    resized_frame = cv2.resize(frame, (64, 64))  # Updated input size
    # Normalize pixel values to range [0, 1]
    normalized_frame = resized_frame.astype('float32') / 255.0
    # Expand dimensions to match model input shape (add batch dimension)
    preprocessed_frame = np.expand_dims(normalized_frame, axis=0)
    return preprocessed_frame

# Function to predict sign language alphabet from webcam frame
def predict_alphabet(frame, bounding_box):
    # Calculate the area of the bounding box
    box_area = (bounding_box[2] - bounding_box[0]) * (bounding_box[3] - bounding_box[1])
    # Define a threshold area to filter out smaller hand signs
    threshold_area = 5000  # Adjust as needed based on your requirements
    
    # Check if the bounding box area is greater than the threshold
    if box_area > threshold_area:
        preprocessed_frame = preprocess_frame(frame)
        # Predict probabilities for each class
        predictions = model.predict(preprocessed_frame)
        # Get the index of the class with the highest probability
        predicted_class_index = np.argmax(predictions)
        # Map the predicted class index to the corresponding alphabet
        alphabet = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 
                    11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 
                    21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}
        predicted_alphabet = alphabet[predicted_class_index]
        return predicted_alphabet
    else:
        return None

def main():
    st.title("Sign Language Recognition")
    
    start_camera = st.button("Start Camera")
    stop_camera = False

    # Create a placeholder for the camera feed
    camera_placeholder = st.empty()

    if start_camera:
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret or stop_camera:
                break

            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect hands in the frame
            results = hands.process(rgb_frame)

            # If hands are detected, make predictions
            if results.multi_hand_landmarks:
                # Get bounding box coordinates
                for hand_landmarks in results.multi_hand_landmarks:
                    x_min = min(hand_landmarks.landmark, key=lambda landmark: landmark.x).x * frame.shape[1]
                    y_min = min(hand_landmarks.landmark, key=lambda landmark: landmark.y).y * frame.shape[0]
                    x_max = max(hand_landmarks.landmark, key=lambda landmark: landmark.x).x * frame.shape[1]
                    y_max = max(hand_landmarks.landmark, key=lambda landmark: landmark.y).y * frame.shape[0]
                    bounding_box = [x_min, y_min, x_max, y_max]
                    # Draw bounding box
                    cv2.rectangle(frame, (int(bounding_box[0]), int(bounding_box[1])), 
                                  (int(bounding_box[2]), int(bounding_box[3])), (0, 255, 0), 2)

                    predicted_alphabet = predict_alphabet(frame, bounding_box)

                    # Display the predicted alphabet if it's not None
                    if predicted_alphabet is not None:
                        # Display the predicted alphabet on the webcam feed
                        cv2.putText(frame, predicted_alphabet, (int(bounding_box[0]), int(bounding_box[1])-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Convert the frame back to BGR after processing
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Display the captured frame in the Streamlit UI
            camera_placeholder.image(bgr_frame)

            # Check for 'q' key to exit loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the camera and close OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

    if start_camera:
        stop_camera = st.button("Stop Camera")

if __name__ == "__main__":
    main()
