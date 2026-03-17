import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader
from src.model_training.model import PlantCNN

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# ── Config — change these to experiment ──────────────────────
EPOCHS      = 20
BATCH_SIZE  = 64
LR          = 0.0001
IMG_SIZE    = 128
NUM_CLASSES = 12

# ── Dataset ───────────────────────────────────────────────────
class PlantDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.tensor(images).permute(0, 3, 1, 2)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# ── Training function ─────────────────────────────────────────
def train_model(model, train_loader, val_loader, epochs, device, criterion, optimizer):
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        train_acc = correct / total
        val_acc   = val_correct / val_total

        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # Log metrics per epoch to MLflow
        mlflow.log_metric("train_loss", train_loss / len(train_loader), step=epoch)
        mlflow.log_metric("val_loss",   val_loss   / len(val_loader),   step=epoch)
        mlflow.log_metric("train_acc",  train_acc,                      step=epoch)
        mlflow.log_metric("val_acc",    val_acc,                        step=epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'src/model/plant_cnn.pth')
            print(f"Epoch {epoch+1}/{epochs} | Train: {train_acc:.3f} | Val: {val_acc:.3f} ✓ saved")
        else:
            print(f"Epoch {epoch+1}/{epochs} | Train: {train_acc:.3f} | Val: {val_acc:.3f}")

    return history, best_val_acc


# ── Main ──────────────────────────────────────────────────────
if __name__ == '__main__':

    # Load data
    images = np.load('data/images_plant.npy').astype('float32') / 255.0
    labels = pd.read_csv('data/Labels_plant.csv')
    le     = LabelEncoder()
    y      = le.fit_transform(labels['Label'])

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(images, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # DataLoaders
    train_loader = DataLoader(PlantDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(PlantDataset(X_val,   y_val),   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(PlantDataset(X_test,  y_test),  batch_size=BATCH_SIZE)

    # Device, model, loss, optimizer
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model     = PlantCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ── MLflow run ────────────────────────────────────────────
    mlflow.set_experiment("PlantCNN")

    with mlflow.start_run():

        # Log parameters
        mlflow.log_param("epochs",      EPOCHS)
        mlflow.log_param("batch_size",  BATCH_SIZE)
        mlflow.log_param("learning_rate", LR)
        mlflow.log_param("img_size",    IMG_SIZE)
        mlflow.log_param("num_classes", NUM_CLASSES)
        mlflow.log_param("optimizer",   "Adam")
        mlflow.log_param("dropout",     0.5)

        # Train
        history, best_val_acc = train_model(
            model, train_loader, val_loader,
            EPOCHS, device, criterion, optimizer
        )

        # Log final summary metrics
        mlflow.log_metric("best_val_acc", best_val_acc)

        # Save model as MLflow artifact
        mlflow.pytorch.log_model(model, "plant_cnn_model")

        # Save the .pth as an artifact too
        mlflow.log_artifact("src/model/plant_cnn.pth")

        print(f"\nRun complete. Best val accuracy: {best_val_acc:.3f}")