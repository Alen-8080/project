import numpy as np
import tflite_runtime.interpreter as tflite
import cv2

# Load your TFLite model
interpreter = tflite.Interpreter(model_path='/home/raspberry/Desktop/converted_tflite/model_unquant.tflite')
interpreter.allocate_tensors()

# Get input details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess your input image
def preprocess_image(image_path):
    input_shape = input_details[0]['shape'][1:3]  # Height and width
    image = cv2.imread(image_path)
    image = cv2.resize(image, input_shape)  # Resize to model's input size
    image = image.astype(np.float32) / 255.0  # Convert to FLOAT32 and normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

input_image_path = '/home/raspberry/Desktop/converted_tflite/test.jpg'
input_image = preprocess_image(input_image_path)

# Set input tensor
interpreter.set_tensor(input_details[0]['index'], input_image)

# Run inference
interpreter.invoke()

# Get output tensor
output_scores = interpreter.get_tensor(output_details[0]['index'])

# Interpret the output (find the class with highest score)
predicted_class_index = np.argmax(output_scores)

class_names = ['fight', 'fire', 'normal']  # Modify this list based on your labels

predicted_class_name = class_names[predicted_class_index]


print(f"Predicted class: {predicted_class_name}")
