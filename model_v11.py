from tflite_runtime.interpreter import Interpreter
import numpy as np
import cv2
import time
import serial

# Load model
model_path = "/home/raspberry/Desktop/converted_tflite/model_unquant.tflite" 
label_path = "/home/raspberry/Desktop/converted_tflite/labels.txt"

interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(label_path, 'r') as f:
    labels = list(map(str.strip, f.readlines()))

# Function to preprocess image
def preprocess_image(frame, size=(224, 224)):
    resized_frame = cv2.resize(frame, size)
    return resized_frame.astype(np.float32) / 255.0

# Function to predict anomaly
def predict_anomaly(frame):
    input_data = np.expand_dims(preprocess_image(frame), axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    probabilities = np.exp(predictions - np.max(predictions)) / np.sum(np.exp(predictions - np.max(predictions)))
    predicted_label = labels[np.argmax(probabilities)]
    return predicted_label, np.max(probabilities)

# Capture video
cap = cv2.VideoCapture(0)  # Change to the appropriate camera index if needed

start_time = time.time()
prev_prediction = "normal"
recording = False
record_start_time = 0
current_prediction = ""

# Setup the serial connection to Arduino
ser = serial.Serial('/dev/ttyACM0', 9600)  # Use the correct port

# Function to send the stop message
def send_stop_message():
    ser.write(b'S')
    print("Send stop signal to Arduino.")

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame from the camera.")
        break

    current_time = time.time()

    # Check if 3 seconds have passed
    if current_time - start_time >= 3:
        current_prediction, _ = predict_anomaly(frame)
        print(f"Current prediction: {current_prediction}")
        
        # Check if both current and previous predictions are not "normal"
        if prev_prediction != "2 normal" and current_prediction != "2 normal":
            if not recording:
                print("Start recording...")
                # Send stop signal to Arduino before recording
                send_stop_message()
                recording = True
                record_start_time = current_time
                video_writer = cv2.VideoWriter('abnormal_event11.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (frame.shape[1], frame.shape[0]))
        else:
            if recording and current_time - record_start_time >= 6:
                print("Stop recording...")
                recording = False
                video_writer.release()

        prev_prediction = current_prediction
        start_time = current_time

    # Record video
    if recording:
        video_writer.write(frame)

    # Display frame
    cv2.putText(frame, f"Anomaly: {current_prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == 27:  # Press ESC to exit
        break

# Close serial connection
ser.close()

# Release camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
