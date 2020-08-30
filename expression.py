import numpy as np
import tensorflow as tf
import cv2
import pathlib
#Set Path
path="face.tflite"
# Load TFLite model and allocate tensors.

interpreter = tf.lite.Interpreter(model_path = path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.allocate_tensors()

# input details
print(input_details)
# output details
print(output_details)


for file in pathlib.Path("D:\\New Folder\\images").iterdir():
    
    
    # Test model on random input data.
    input_shape = input_details[0]['shape']
    print(input_shape)
    print(input_details[0]['index'])
    print(output_details[0]['index'])
      
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
     # run the inference
    interpreter.invoke()
    # The function `get_tensor()` returns a copy of the tensor data.
    output_data = interpreter.get_tensor(output_details[0]['index'])

    print("For file {}, the output is {}".format(file.stem, output_data))
   