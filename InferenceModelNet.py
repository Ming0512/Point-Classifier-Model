import numpy as np
import os
import torch
import torch.nn as nn

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

class ToTensor(object):
    def __call__ (self, points):
        return torch.from_numpy(points).float()

# Load your own point cloud sample (shape: [2048, 3])
data_path = r"C:\VsCode\AI&ML\ModelNet\data"

points_test = np.load(os.path.join(data_path,'points_test.npy'))
labels_test = np.load(os.path.join(data_path,'labels_test.npy'))

# First 10 sample points
sample_points = points_test[:100]
sample_labels = labels_test[:100]

inference_transform = ToTensor()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PointNetClassifier().to(device)

model.load_state_dict(torch.load("best_model.pth", map_location = device))
model.eval()

sample_points = inference_transform(sample_points)

with torch.no_grad():
    output = model(sample_points)
    predicted_class = output.argmax(dim=1)

for i in range(len(sample_points)):
    print(f"Sample {i+1}: Predicted class: {predicted_class[i].item()}, Ground Truth class: {sample_labels[i]}")
