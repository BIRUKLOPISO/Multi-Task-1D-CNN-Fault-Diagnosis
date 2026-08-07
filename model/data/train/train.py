import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from model.model import MultiTask1DCNN
from tqdm import tqdm
import os

class FaultDataset(Dataset):
    def __init__(self, X, y_type, y_section):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_type = torch.tensor(y_type, dtype=torch.long)
        self.y_section = torch.tensor(y_section, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_type[idx], self.y_section[idx]

print("Loading real dataset...")
X = np.load("data/dataset_real/X.npy")
y_type = np.load("data/dataset_real/y_type.npy")
y_section = np.load("data/dataset_real/y_section.npy")

mean = X.mean(axis=(0, 2), keepdims=True)
std = X.std(axis=(0, 2), keepdims=True) + 1e-8
X = (X - mean) / std

dataset = FaultDataset(X, y_type, y_section)

train_size = int(0.70 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_set, val_set, test_set = random_split(
    dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = MultiTask1DCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 40
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for Xb, yt, ys in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        Xb, yt, ys = Xb.to(device), yt.to(device), ys.to(device)

        optimizer.zero_grad()
        type_out, section_out = model(Xb)
        loss = criterion(type_out, yt) + criterion(section_out, ys)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for Xb, yt, ys in val_loader:
            Xb, yt, ys = Xb.to(device), yt.to(device), ys.to(device)
            type_out, section_out = model(Xb)
            val_loss += (criterion(type_out, yt) + criterion(section_out, ys)).item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        os.makedirs("saved_models", exist_ok=True)
        torch.save(model.state_dict(), "saved_models/best_model_real.pth")
        print("  → Best model saved")

print("Training finished.")
