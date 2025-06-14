import numpy as np
import pandas as pd
import kagglehub
# Download latest version
path = kagglehub.dataset_download("ruhulaminsharif/eye-disease-image-dataset")
print("Path to dataset files:", path)
import os
import time
import random
import copy  # <-- Add this import!
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import transforms, datasets
import torchvision.models as models
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report
from torchvision.models import VGG16_Weights, MobileNet_V3_Large_Weights, DenseNet121_Weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Define dataset path
#dataset_path = "/kaggle/input/eye-disease-image-dataset"
dataset_path = path
# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
# Load the dataset
full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
# Define train/val/test split ratios
train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1
total_size = len(full_dataset)
train_size = int(train_ratio * total_size)
val_size = int(val_ratio * total_size)
test_size = total_size - train_size - val_size  # Ensure total sums up correctly

# Split dataset
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Number of classes
num_classes = len(full_dataset.classes)
print(f"Classes: {full_dataset.classes}, Total Classes: {num_classes}")

# Function to count images per class
def count_images_per_class(dataset):
    class_counts = {cls: 0 for cls in dataset.dataset.classes}
    for _, label in dataset.dataset.samples:
        class_counts[dataset.dataset.classes[label]] += 1
    return class_counts

# Get the count of images per class
class_counts = count_images_per_class(train_dataset)

# Print the count of images per class
for class_name, count in class_counts.items():
    print(f"Class '{class_name}': {count} images")
	
	train_class_counts = count_images_per_class(train_dataset)

# Generate a list of colors based on the number of classes
colors = plt.cm.viridis(np.linspace(0, 1, len(train_class_counts)))

# Plot class distribution with different colors
plt.figure(figsize=(13, 5))
plt.bar(train_class_counts.keys(), train_class_counts.values(), color=colors)
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.title("Class Distribution in Training Dataset")
plt.xticks(rotation=45)
plt.show()

def show_random_images(dataset, num_images=1):
    # Create a grid with 5 images per row
    rows = len(dataset.dataset.classes) // 5 + (len(dataset.dataset.classes) % 5 > 0)
    fig, axes = plt.subplots(rows, 5, figsize=(20, 5 * rows))
    # Flatten axes in case the grid is not fully filled (when the number of classes is not a multiple of 5)
    axes = axes.flatten()
    for idx, cls in enumerate(dataset.dataset.classes):
        # Get the indices for images of the current class
        class_indices = [i for i, (_, label) in enumerate(dataset.dataset.samples) if dataset.dataset.classes[label] == cls]

        # Choose a random index for the class
        random_idx = random.choice(class_indices)
        img_path, _ = dataset.dataset.samples[random_idx]

        # Open and display the image
        img = Image.open(img_path)
        axes[idx].imshow(img)
        axes[idx].axis("off")
        axes[idx].set_title(cls)
    # Hide any unused axes (in case the number of classes is not divisible by 5)
    for i in range(len(dataset.dataset.classes), len(axes)):
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()

# Show one random image per class
show_random_images(train_dataset)

def get_model_mobilenetv3(num_classes, freeze_layers=True):
    # Load MobileNetV3 with pre-trained weights
    model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)

    if freeze_layers:
        # Freeze all blocks in model.features except for the last two modules
        total_blocks = len(model.features)
        for idx, module in enumerate(model.features):
            if idx < total_blocks - 2:
                for param in module.parameters():
                    param.requires_grad = False
            else:
                # Unfreeze last two blocks
                for param in module.parameters():
                    param.requires_grad = True
    # Replace classifier with a custom head that includes dropout
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )
    return model.to(device)
	
def get_model_densenet121(num_classes, freeze_layers=True):
    # Load DenseNet121 with pre-trained weights
    model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)

    if freeze_layers:
        # For DenseNet121, model.features is a Sequential of various layers.
        # Unfreeze only the last two modules.
        features = list(model.features.children())
        total_blocks = len(features)
        for idx, module in enumerate(features):
            if idx < total_blocks - 2:
                for param in module.parameters():
                    param.requires_grad = False
            else:
                for param in module.parameters():
                    param.requires_grad = True
        # Note: The model.features remains a Sequential container.
    # Replace classifier with dropout and a linear layer.
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.classifier.in_features, num_classes)
    )
    return model.to(device)
	
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0

    def should_stop(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience
		
def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, early_stopping, epochs=20):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    all_val_labels, all_val_preds = [], []
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(epochs):
        start_time = time.time()
        # Training phase
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            preds = outputs.argmax(1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
        train_loss = running_train_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                preds = outputs.argmax(1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(preds.cpu().numpy())
        val_loss = running_val_loss / len(val_loader)
        val_acc = correct_val / total_val
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        scheduler.step(val_loss)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_time:.2f}s")
        if early_stopping.should_stop(val_loss):
            print("Early stopping triggered.")
            break
        if val_loss < early_stopping.best_loss:
            best_model_wts = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model_wts)
    return model, train_losses, val_losses, train_accs, val_accs, all_val_labels, all_val_preds
	
def plot_accuracy_and_loss(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12, 5))
    # Accuracy curve
    plt.subplot(1, 2, 1)
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(True)
    # Loss curve
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
	
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
	
def plot_per_class_accuracy(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    per_class_accuracy = np.diag(cm) / cm.sum(axis=1)
    plt.figure(figsize=(12, 6))
    plt.bar(class_names, per_class_accuracy, color="skyblue")
    plt.xlabel("Classes")
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy")
    plt.xticks(rotation=45)
    plt.show()
	
class_names = full_dataset.classes
print("Classes:", class_names)
num_classes = len(class_names)  # ensure class_names is defined from your dataset
epochs = 20
learning_rate = 0.001
print("\nTraining MobileNetV3 (unfreeze last 2 blocks) with Adam optimizer...\n")
mobilenet_model = get_model_mobilenetv3(num_classes, freeze_layers=True)
optimizer_mnv3 = optim.Adam(mobilenet_model.parameters(), lr=learning_rate)
scheduler_mnv3 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_mnv3, mode='min', factor=0.1, patience=3)
early_stopping_mnv3 = EarlyStopping(patience=5)
mobilenet_model, train_losses_mnv3, val_losses_mnv3, train_accs_mnv3, val_accs_mnv3, val_labels_mnv3, val_preds_mnv3 = train_model(
    mobilenet_model, nn.CrossEntropyLoss(), optimizer_mnv3, scheduler_mnv3,
    train_loader, val_loader, early_stopping_mnv3, epochs=epochs
)
print("\nMobileNetV3 Training Completed.\n")
plot_accuracy_and_loss(train_losses_mnv3, val_losses_mnv3, train_accs_mnv3, val_accs_mnv3)
plot_confusion_matrix(val_labels_mnv3, val_preds_mnv3, class_names)
plot_per_class_accuracy(val_labels_mnv3, val_preds_mnv3, class_names)
