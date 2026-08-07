import sys
sys.path.append(".")

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from model.model import MultiTask1DCNN
from sklearn.metrics import accuracy_score, f1_score, classification_report

class FaultDataset(Dataset):
    def __init__(self, X, y_type, y_section):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_type = torch.tensor(y_type, dtype=torch.long)
        self.y_section = torch.tensor(y_section, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_type[idx], self.y_section[idx]

# Load data
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

_, _, test_set = random_split(dataset, [train_size, val_size, test_size],
                              generator=torch.Generator().manual_seed(42))

test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTask1DCNN().to(device)
model.load_state_dict(torch.load("saved_models/best_model_real.pth", map_location=device))
model.eval()

all_type_true, all_type_pred = [], []
all_section_true, all_section_pred = [], []

with torch.no_grad():
    for Xb, yt, ys in test_loader:
        Xb = Xb.to(device)
        type_out, section_out = model(Xb)
        all_type_true.extend(yt.numpy())
        all_type_pred.extend(type_out.argmax(1).cpu().numpy())
        all_section_true.extend(ys.numpy())
        all_section_pred.extend(section_out.argmax(1).cpu().numpy())

print("="*55)
print("TEST RESULTS")
print("="*55)
print(f"Fault Type Accuracy : {accuracy_score(all_type_true, all_type_pred)*100:.2f}%")
print(f"Fault Type F1-score : {f1_score(all_type_true, all_type_pred, average='macro')*100:.2f}%")
print(f"Section Accuracy    : {accuracy_score(all_section_true, all_section_pred)*100:.2f}%")
print(f"Section F1-score    : {f1_score(all_section_true, all_section_pred, average='macro')*100:.2f}%")
print("="*55)
print("\nFault Type Classification Report:")
print(classification_report(all_type_true, all_type_pred, digits=4))
