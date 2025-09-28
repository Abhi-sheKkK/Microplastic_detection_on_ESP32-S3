import tensorflow as tf

# Path to your .tflite file
model_path = "model_quant.tflite"

# Load the TFLite model
with open(model_path, "rb") as f:
    model_content = f.read()

# Parse the model
interpreter = tf.lite.Interpreter(model_content=model_content)
interpreter.allocate_tensors()  # Ensures model validity

# Get unique ops
ops = set()
for op in interpreter._get_ops_details():
    ops.add(op['op_name'])

# Print results
print("Required ops in the model:")
for op in sorted(ops):
    print(op)
print(f"Total unique ops: {len(ops)}")