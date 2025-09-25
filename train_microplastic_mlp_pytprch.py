import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix, classification_report
import seaborn as sns

epochs = 400 
WAVE_LENGTH = 200 
WAVE_COLS = [f"v{i}" for i in range(WAVE_LENGTH)] 

# Load generated synthetic dataset
CSV_PATH = "Microplastic_detection_on_ESP32-S3\synthetic_microplastics_1s_200pts_V2.csv"
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print(f"Error: The file '{CSV_PATH}' was not found.")
    exit()

# Data Preparation 
feature_cols = ["peak_count", "mean_peak_width_ms", "mean_peak_energy", "mean_peak_symmetry",
                "mean_rise_ms", "mean_fall_ms", "total_energy"]
label_col = "label"

X_wave = df[WAVE_COLS].values.astype(np.float32)
X_feat = df[feature_cols].values.astype(np.float32)
y_reg = df[label_col].values.astype(np.float32)
y_stratify = df[label_col].values.astype(np.int64)

# Normalize features
from sklearn.preprocessing import StandardScaler
scaler_feat = StandardScaler()
X_feat = scaler_feat.fit_transform(X_feat)

scaler_wave = StandardScaler()
X_wave = scaler_wave.fit_transform(X_wave)

# Check for data imbalance
unique, counts = np.unique(y_stratify, return_counts=True)
print("Class distribution in the full dataset:")
for label, count in zip(unique, counts):
    print(f"Label {label}: {count} samples")

# Train-test split
X_wave_train, X_wave_test, X_feat_train, X_feat_test, y_reg_train, y_reg_test = train_test_split(
    X_wave, X_feat, y_reg, test_size=0.2, random_state=42, stratify=y_stratify
)

# Convert to PyTorch Tensors
X_wave_train = torch.tensor(X_wave_train).unsqueeze(1) 
X_wave_test = torch.tensor(X_wave_test).unsqueeze(1)
X_feat_train = torch.tensor(X_feat_train)
X_feat_test = torch.tensor(X_feat_test)
y_train = torch.tensor(y_reg_train).unsqueeze(1)
y_test = torch.tensor(y_reg_test).unsqueeze(1)

wave_len = X_wave.shape[1]
feat_dim = X_feat.shape[1]
num_classes = len(np.unique(y_stratify)) 

# CNN-Regressor
class CNN_Regressor(nn.Module):
    def __init__(self, wave_len, feat_dim, max_count=8):
        super(CNN_Regressor, self).__init__()
        
        self.max_count = max_count # For Sigmoid scaling
        
        # 1D Convolutional Layers 
        self.conv_layers = nn.Sequential(
            # Layer 1: (1, L) -> (16, L/2)
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            # Layer 2: (16, L/2) -> (32, L/4)
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            # Layer 3: NEW LAYER (32, L/4) -> (64, L/8)
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        conv_output_size = self._get_conv_output_size(wave_len)
        
        # Final MLP Layers (Combined Input)
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size + feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Single output for regression
        )
    
    def _get_conv_output_size(self, wave_len):
        x = torch.randn(1, 1, wave_len)
        x = self.conv_layers(x)
        return x.flatten().shape[0]

    def forward(self, x_wave, x_feat):
        x_wave = self.conv_layers(x_wave)
        x_wave = x_wave.view(x_wave.size(0), -1) 
        
        combined = torch.cat((x_wave, x_feat), dim=1)
        
        # Final output uses a Sigmoid scaled by max_count to constrain the prediction, 
        # helping the model avoid wildly high predictions and stabilize near zero.
        raw_output = self.fc_layers(combined)
        return torch.sigmoid(raw_output) * self.max_count

# Instantiate the model with max count 8 
model = CNN_Regressor(wave_len, feat_dim, max_count=8) 

# Criterion: MSELoss 
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001)

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_exact_accuracy(y_true, y_pred_raw):
    y_pred_int = torch.round(y_pred_raw).int().squeeze()
    y_true_int = y_true.int().squeeze()
    correct = (y_pred_int == y_true_int).sum().item()
    total = y_true_int.size(0)
    return correct / total

# Training setup
train_losses = []
test_losses = []
train_rmse = []
test_rmse = []
train_accuracies = [] 
test_accuracies = []

# Training loop
print("\n--- Starting OPTIMIZED CNN-Regressor Training ---")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_wave_train, X_feat_train)
    loss = criterion(outputs, y_train) 
    loss.backward()
    optimizer.step()
    
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_wave_test, X_feat_test)
        test_loss = criterion(test_outputs, y_test)
        
        test_rmse_val = calculate_rmse(y_test.cpu().numpy(), test_outputs.cpu().numpy())
        test_acc_val = calculate_exact_accuracy(y_test, test_outputs)
        train_rmse_val = calculate_rmse(y_train.cpu().numpy(), outputs.cpu().numpy())
        train_acc_val = calculate_exact_accuracy(y_train, outputs)
    
    train_losses.append(loss.item())
    test_losses.append(test_loss.item())
    train_rmse.append(train_rmse_val)
    test_rmse.append(test_rmse_val)
    train_accuracies.append(train_acc_val)
    test_accuracies.append(test_acc_val) 
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss (MSE): {loss.item():.4f}, Train RMSE: {train_rmse_val:.4f}, Test Loss (MSE): {test_loss.item():.4f}, Test RMSE: {test_rmse_val:.4f}, Test Acc (Rounded): {test_acc_val:.4f}")

# Plot Loss Curves
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train MSE Loss', color='blue')
plt.plot(test_losses, label='Test MSE Loss', color='red')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss (MSE)", fontsize=12)
plt.title("Training and Test Loss Curves", fontsize=16)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Accuracy Curves 
plt.figure(figsize=(10, 6))
plt.plot(train_accuracies, label='Train Exact Match Accuracy', color='blue')
plt.plot(test_accuracies, label='Test Exact Match Accuracy', color='red')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Exact Match Accuracy (Rounded)", fontsize=12)
plt.title("Training and Test Accuracy Curves", fontsize=16)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot RMSE Curves
plt.figure(figsize=(10, 6))
plt.plot(train_rmse, label='Train RMSE', color='darkgreen')
plt.plot(test_rmse, label='Test RMSE', color='orange')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Root Mean Squared Error", fontsize=12)
plt.title("Training and Test RMSE Curves", fontsize=16)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()

# --- FINAL INFERENCE AND EVALUATION ---
model.eval()
with torch.no_grad():
    predictions_raw = model(X_wave_test, X_feat_test) 
    
    # Final Integer Predictions and True Labels
    predictions_int = torch.round(predictions_raw).int().squeeze()
    y_test_int = y_test.int().squeeze() 
    
    # Print sample inference results
    print("\nSample Inference Results (True vs Predicted):")
    for i in range(10):
        print(f"True: {y_test_int[i].item()}, Predicted: {predictions_int[i].item()}")

    # Final overall accuracy (Exact Match)
    final_accuracy = accuracy_score(y_test_int, predictions_int)
    print(f"\nFinal Test Accuracy (Exact Match after Rounding): {final_accuracy:.4f}")
    
    # Calculate Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)
    mae = mean_absolute_error(y_test_int.cpu().numpy(), predictions_int.cpu().numpy())
    rmse = calculate_rmse(y_test_int.cpu().numpy(), predictions_int.cpu().numpy())
    print(f"Mean Absolute Error (MAE): {mae:.4f} (Average Count Error)")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test_int, predictions_int)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(y_test_int.cpu().numpy()), yticklabels=np.unique(y_test_int.cpu().numpy()))
    plt.title("Confusion Matrix of Rounded Regression Predictions", fontsize=16)
    plt.ylabel("True Label (Count)", fontsize=12)
    plt.xlabel("Predicted Label (Rounded Count)", fontsize=12)
    plt.tight_layout()
    plt.show()

    # Print Classification Report
    print("\nClassification Report (After Rounding):")
    all_labels = np.unique(np.concatenate((y_test_int.cpu().numpy(), predictions_int.cpu().numpy())))
    print(classification_report(y_test_int, predictions_int, target_names=[str(i) for i in all_labels], zero_division=0))

save_path = "Microplastic_detection_on_ESP32-S3\cnn_regressor_model.pth"
torch.save(model.state_dict(), save_path)
