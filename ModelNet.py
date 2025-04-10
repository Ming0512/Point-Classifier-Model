import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# TODO:Task 1: Import necessary libraries:
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim


# set seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# visualize point clouds for a batch
def visualize_point_cloud(batch, fig_size=(20, 20)):
    if torch.is_tensor(batch):
        batch = batch.numpy()

    fig = plt.figure(figsize=fig_size)
    for i in range(16):
        point_cloud = batch[i]
        row = i // 4
        col = i % 4
        # Create 3D subplot
        ax = fig.add_subplot(4, 4, i+1, projection='3d')

        # Extract x, y, z coordinates
        x = point_cloud[:, 0]
        y = point_cloud[:, 1]
        z = point_cloud[:, 2]
        # Plot 3D scatter
        ax.scatter(x, y, z, s=1, alpha=0.5)

        ax.set_title(f'Point Cloud {i+1}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])
        ax.grid(False)

    plt.tight_layout()
    plt.savefig('visualize.jpg', format='jpg',bbox_inches='tight')

# Point Cloud Transforms
class RandomRotation(object):
    def __call__(self, points):
        # Random rotation around z-axis
        theta = np.random.uniform(0, 2*np.pi)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        return np.dot(points, rotation_matrix)

class RandomJitter(object):
    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip
        
    def __call__(self, points):
        jitter = np.clip(np.random.normal(0, self.sigma, points.shape), -self.clip, self.clip)
        return points + jitter

class ToTensor(object):
    def __call__(self, points):
        return torch.from_numpy(points).float()


# Custom ModelNet10 Dataset
class ModelNet10(Dataset):
    def __init__(self, data_path, phase='train', num_points=2048, transforms=None):
        self.num_points = num_points
        self.transforms = transforms

        # Load the point cloud and label files
        if phase == 'train':
            self.points = np.load(os.path.join(data_path, 'points_train.npy'))  # Shape: (1500, 2048, 3)
            self.labels = np.load(os.path.join(data_path, 'labels_train.npy'))  # Shape: (1500, 1)
        else:
            self.points = np.load(os.path.join(data_path, 'points_test.npy'))   # Shape: (250, 2048, 3)
            self.labels = np.load(os.path.join(data_path, 'labels_test.npy'))   # Shape: (250, 1)

        # Ensure the number of labels matches the number of point clouds
        assert len(self.points) == len(self.labels), "Mismatch between points and labels"


    def __getitem__(self, idx):

        points = self.points[idx]  # Shape: (2048, 3)
        label = int(self.labels[idx][0])  # Extract label as integer

        # Randomly sample `num_points` if the cloud has more points
        if points.shape[0] > self.num_points:
            idxs = np.random.choice(points.shape[0], self.num_points, replace=False)
            points = points[idxs]

        # Apply transformations if provided
        if self.transforms:
            points = self.transforms(points)

        return points, label

    def __len__(self):
        return len(self.points)

# Define PointNetClassifier model
class PointNetClassifier(nn.Module):
    def __init__(self, num_channel=3, num_classes=10):
        super(PointNetClassifier, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(num_channel,64,kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.global_maxpool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1)  
        )
    def forward(self, x):
        # Reshape: (B, N, 3) → (B, 3, N) for Conv1d
        x = x.permute(0, 2, 1)

        x = self.conv1(x)      # (B, 3, N) → (B, 64, N)
        x = self.conv2(x)      # (B, 64, N) → (B, 128, N)
        x = self.conv3(x)      # (B, 128, N) → (B, 1024, N)
        x = self.global_maxpool(x)  # (B, 1024, N) → (B, 1024, 1)
        x = x.view(x.size(0), -1)   # Flatten: (B, 1024)
        

        x = self.fc1(x)        # (B, 1024) → (B, 512)
        x = self.fc2(x)        # (B, 512) → (B, 256)
        x = self.fc3(x)        # (B, 256) → (B, 10)

        return x

# Training Function
def train(model, train_loader, optimizer, criterion, device='cpu'):
    model.train()
    total_loss = 0
    correct = 0
    total_samples = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        inputs, labels = inputs.to(device), labels.to(device)
        # forward pass
        outputs = model(inputs)
        # Compute loss
        loss = criterion(outputs,labels)
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        # # Print every 10 batches to monitor progress
        if batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1:
            print(f"Batch {batch_idx}/{len(train_loader)-1}, Loss: {loss.item():.4f}")

    # Compute average loss and accuracy
    avg_loss = total_loss/len(train_loader)
    accuracy = correct/ total_samples

    return avg_loss, accuracy

# Evaluation function
def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            total_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    # Average accuracy
    avg_loss = total_loss/len(val_loader)
    accuracy = correct/total_samples
    return avg_loss, accuracy

def main():
    # 'Path to ModelNet10 dataset'
    data_path = r"\VsCode\AI&ML\ModelNet\data"
    epochs = 50
    lr = 1e-3

    batch_size = 16
    # Number of points in each point cloud
    num_points = 2048
    seed = 42
    set_seed(seed)

    # Force CPU usage
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Data transforms custom
    train_transform = transforms.Compose([
        RandomRotation(), 
        RandomJitter(), 
        ToTensor()
    ])


    test_transform = transforms.Compose([
        ToTensor()
    ])


    # Load dataset and dataloader :
    train_dataset = ModelNet10(data_path, phase ='train',num_points =num_points, transforms= train_transform)
    test_dataset = ModelNet10(data_path, phase= 'test', num_points= num_points, transforms =test_transform)

    train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle= True)
    test_loader =  DataLoader(test_dataset, batch_size= batch_size, shuffle = False)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")

    for batch, _ in train_loader:
        visualize_point_cloud(batch)
        break

    # Initialize model
    model = PointNetClassifier().to(device)
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)


    # Training loop
    best_acc = 0.0
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved best model!")
    
    print(f"Best validation accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()
