import torch
import torch.nn as nn
import onnx
import tensorflow as tf
import numpy as np
import os
import onnx2tf

# Disable GPU for TensorFlow to avoid CUDA errors
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Define the original PyTorch model (unchanged)
class CNN_Regressor(nn.Module):
    def __init__(self, wave_len, feat_dim, max_count=8):
        super(CNN_Regressor, self).__init__()
        self.max_count = max_count
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        conv_output_size = self._get_conv_output_size(wave_len)
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size + feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def _get_conv_output_size(self, wave_len):
        x = torch.randn(1, 1, wave_len)
        x = self.conv_layers(x)
        print(f"Conv output shape: {x.shape}")  # Debug print, expected [1, 64, 25]
        return x.flatten().shape[0]

    def forward(self, x_wave, x_feat):
        x_wave = self.conv_layers(x_wave)
        x_wave = torch.reshape(x_wave, (x_wave.size(0), -1))  # Explicit reshape
        combined = torch.cat((x_wave, x_feat), dim=1)
        raw_output = self.fc_layers(combined)
        return torch.sigmoid(raw_output) * self.max_count

# Load the model
model = CNN_Regressor(wave_len=200, feat_dim=7, max_count=8)
model.load_state_dict(torch.load("./cnn_regressor_model.pth"))
model.eval()

# Export to ONNX with fixed batch size
dummy_wave = torch.randn(1, 1, 200)
dummy_feat = torch.randn(1, 7)
torch.onnx.export(
    model,
    (dummy_wave, dummy_feat),
    "model.onnx",
    input_names=["wave_input", "feat_input"],
    output_names=["output"],
    opset_version=13
)

# Verify ONNX model
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model is valid")

# Convert ONNX to TensorFlow SavedModel using onnx2tf
tf_model_path = "model_tf"
onnx2tf.convert(
    input_onnx_file_path="model.onnx",
    output_folder_path=tf_model_path,
    output_signaturedefs=True,  # Generate signature for TFLite conversion
    copy_onnx_input_output_names_to_tflite=True
)

# Representative dataset for quantization
def representative_dataset():
    for _ in range(100):
        yield {
            "wave_input": np.random.randn(1, 1, 200).astype(np.float32),
            "feat_input": np.random.randn(1, 7).astype(np.float32)
        }

# Convert TensorFlow SavedModel to TFLite with int8 quantization
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.experimental_new_quantizer = True

try:
    tflite_model = converter.convert()
except Exception as e:
    print(f"TFLite conversion failed: {e}")
    raise

# Save the quantized TFLite model
tflite_path = "model_quant_int8.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

# Check model size
print(f"Int8 TFLite model size: {os.path.getsize(tflite_path)} bytes")