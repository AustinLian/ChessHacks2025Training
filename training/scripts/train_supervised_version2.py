import sys
import os

# Add project root to sys.path
# This allows importing training.models.* from a script inside training/scripts/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet_policy_value import ResNetPolicyValue
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os

# -------------------------------
# Dummy dataset (replace with real data!)

# -------------------------------
class ChessDataset(Dataset):
    def __init__(self, X, Y_policy, Y_value):
        self.X = X
        self.Y_policy = Y_policy
        self.Y_value = Y_value

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y_policy[idx], self.Y_value[idx]

# -------------------------------
# Configuration
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
INPUT_PLANES = 18
POLICY_SIZE = 20480
CHANNELS = 128
NUM_BLOCKS = 10
CHECKPOINT_DIR = "checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -------------------------------
# Create model
# -------------------------------
model = ResNetPolicyValue(
    num_blocks=NUM_BLOCKS,
    channels=CHANNELS,
    input_planes=INPUT_PLANES,
    policy_channels=32,
    value_channels=32,
    policy_size=POLICY_SIZE
).to(DEVICE)

# -------------------------------
# Optimizer & loss
# -------------------------------
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_policy = nn.CrossEntropyLoss()
loss_value = nn.MSELoss()

# -------------------------------
# Load dataset
# -------------------------------
# TODO: Replace with real dataset loading
# Example: np.load("training/data/processed/sf_supervised_dataset2024.npz")
# X -> (N, 18, 8, 8)
# Y_policy -> (N,) indices 0..20479
# Y_value -> (N,1)
X = torch.randn((1000, 18, 8, 8), dtype=torch.float32)
Y_policy = torch.randint(0, POLICY_SIZE, (1000,), dtype=torch.long)
Y_value = torch.rand((1000, 1), dtype=torch.float32)

dataset = ChessDataset(X, Y_policy, Y_value)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -------------------------------
# Training loop
# -------------------------------
for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0
    for x_batch, y_policy_batch, y_value_batch in train_loader:
        x_batch = x_batch.to(DEVICE)
        y_policy_batch = y_policy_batch.to(DEVICE)
        y_value_batch = y_value_batch.to(DEVICE)

        optimizer.zero_grad()
        pred_policy, pred_value = model(x_batch)
        loss = loss_policy(pred_policy, y_policy_batch) + loss_value(pred_value, y_value_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}/{EPOCHS} - Avg Loss: {avg_loss:.4f}")

    # Save checkpoint
    torch.save({
        "model_state_dict": model.state_dict()
    }, os.path.join(CHECKPOINT_DIR, "best_model.pt"))

print("✅ Training finished and model saved to checkpoints/best_model.pt")
