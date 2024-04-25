import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import pandas as pd

data = pd.read_csv('ShopeeData_cleaned_sor.csv')

data.drop(columns=['historical_sold'], inplace=True)

features = data.drop(columns=['num_sold', 'Date'])

target = data['num_sold']

# Step 1: Data Preparation
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)

def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i-n_steps:i, :-1])  # all features except target
        y.append(data[i, -1])  # target feature
    return np.array(X), np.array(y)

n_steps = 3
data = np.hstack((scaled_features, target.values.reshape(-1, 1)))
X, y = create_sequences(data, n_steps)

# Convert numpy arrays to PyTorch tensors
X_train, X_test, y_train, y_test = map(
    torch.tensor, (X[:int(len(X) * 0.8)], X[int(len(X) * 0.8):], y[:int(len(y) * 0.8)], y[int(len(y) * 0.8):])
)

train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Step 2: Build the RNN Model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0, c0 = (torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device),
                torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device))
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) # we only want the last time step output
        return out

model = LSTMModel(input_dim=X.shape[2], hidden_dim=50, num_layers=1, output_dim=1)
model = model.float()

# Step 3: Compile and Train the Model
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)

def train_model(model, train_loader):
    model.train()
    for epoch in range(50):
        for seq, labels in train_loader:
            optimizer.zero_grad()
            seq, labels = seq.float(), labels.float()
            y_pred = model(seq)
            loss = criterion(y_pred.squeeze(), labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1} Loss: {loss.item()}')

train_model(model, train_loader)

# Optionally, evaluate the model

model.eval()
with torch.no_grad():
    predictions = model(X_test.float())
    test_loss = criterion(predictions.squeeze(), y_test.float())
    print(f'Test MSE: {test_loss.item()}')
    test_rmse = torch.sqrt(test_loss)
    print(f'Test RMSE: {test_rmse.item()}')
# Duy
# Lê Văn Duy
# input_dim: Kích thước của từng mẫu dữ liệu đầu vào.
# hidden_dim: Số đơn vị trong lớp ẩn của LSTM.
# num_layers: Số lớp LSTM xếp chồng lên nhau.