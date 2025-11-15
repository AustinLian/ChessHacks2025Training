import sys
import torch
sys.path.append(r"F:/VS Code Storage/ChessHacks2025/training/models")
from resnet_policy_value import ResNetPolicyValue

# -------------------------------
# Load model
# -------------------------------
model = ResNetPolicyValue(
    num_blocks=10,       # match training
    channels=128,
    input_planes=18,     # your input planes
    policy_channels=32,
    value_channels=32,
    policy_size=20480     # total moves
)

checkpoint = torch.load("F:/VS Code Storage/ChessHacks2025/checkpoints/best_model.pt")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# -------------------------------
# Dummy board input (batch size = 1)
# -------------------------------
dummy_board = torch.zeros((1, 18, 8, 8))  # 18 planes, 8x8 board

policy, value = model.predict(dummy_board)

print("✅ Model loaded successfully!")
print("Policy output shape:", policy.shape)   # (1, 4672)
print("Value output shape:", value.shape)     # (1, 1)
print("Policy sum:", policy.sum())  # should be 1

# -------------------------------
# Optional: pick move
# -------------------------------
# Choose move index with highest probability
move_index = torch.argmax(policy, dim=-1).item()
print("Predicted move index:", move_index)
