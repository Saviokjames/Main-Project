import torch.onnx
from transformers import pipeline
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

# Load the OCR model
ocr = pipeline('image-to-text', model="microsoft/trocr-base-handwritten")

# Define an example input
example_input = {
    'image': 'D:/Huggingface/this.jpeg'
}

# Export the PyTorch-based OCR model to ONNX
torch.onnx.export(ocr.model, example_input, "ocr_model.onnx", input_names=['image'], output_names=['text'])

# Load the exported ONNX model
onnx_model = onnx.load("ocr_model.onnx")

# Convert the ONNX model to TensorFlow
tf_rep = prepare(onnx_model)
tf_rep.export_graph("ocr_model.pb")

# Convert the TensorFlow model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_saved_model("ocr_model.pb")
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open("ocr_model.tflite", "wb") as f:
    f.write(tflite_model)
