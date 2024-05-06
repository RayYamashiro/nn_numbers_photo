import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor
from sklearn.metrics import confusion_matrix, accuracy_score

train_data = torchvision.datasets.MNIST(root='./data', train=True,
transform=ToTensor(), download=True)
test_data = torchvision.datasets.MNIST(root='./data', train=False,
transform=ToTensor(), download=True)
train_data, val_data = random_split(train_data, [50000, 10000])
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

class MLP(nn.Module):
 def __init__(self):
  super(MLP, self).__init__()
  self.fc1 = nn.Linear(28*28, 128)
  self.fc2 = nn.Linear(128, 10)

 def forward(self, x):
  x = x.view(-1, 28*28)
  x = F.relu(self.fc1(x))
  x = F.softmax(self.fc2(x), dim=1)
  return x

model = MLP()
optimizer = optim.SGD(model.parameters(), lr=0.01)


def train_epoch(model, data_loader, optimizer, loss_fn):
  model.train()
  for X, Y in data_loader:
    optimizer.zero_grad()
    Y_pred = model(X)
    loss = loss_fn(Y_pred, Y)
    loss.backward()
    optimizer.step()

def evaluate_model(model, data_loader, loss_fn):
 model.eval()
 total_loss = 0
 with torch.no_grad():
  for X, Y in data_loader:
    Y_pred = model(X)
    loss = loss_fn(Y_pred, Y)
    total_loss += loss.item()
 return total_loss / len(data_loader)

num_epochs = 20
loss_fn = nn.CrossEntropyLoss()
for epoch in range(num_epochs):
 train_epoch(model, train_loader, optimizer, loss_fn)
 val_loss = evaluate_model(model, val_loader, loss_fn)
 print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")


def predict(model, loader):
 model.eval()
 all_preds = []
 with torch.no_grad():
  for X, _ in loader:
    preds = model(X)
    _, pred_labels = torch.max(preds, 1)
    all_preds.extend(pred_labels.numpy())
 return all_preds


Y_pred = predict(model, test_data)
accuracy = accuracy_score(test_data.targets.numpy(), Y_pred)
cm = confusion_matrix(test_data.targets.numpy(), Y_pred)
print(f"Точность: {accuracy:.4f}")
print("Матрица ошибок:")
print(cm)


def visualize_predictions(images, labels, preds, num_samples=10):
 idxs = np.random.choice(len(images), size=num_samples, replace=False)
 fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
 for i, ax in zip(idxs, axes):
  ax.imshow(images[i].numpy().squeeze(), cmap='gray')
  ax.set_title(f"{preds[i]} (true: {labels[i]})")
  ax.axis('off')
 plt.show()


test_images, test_labels = next(iter(test_loader))
print(test_labels)
model.eval()
all_preds = []
with torch.no_grad():
 for X in test_images:
  preds = model(X)
  _, pred_labels = torch.max(preds, 1)
  all_preds.extend(pred_labels.numpy())
visualize_predictions(test_images, test_labels.numpy(), all_preds)


def preprocess_image(image_path, target_size=(28, 28)):
    # Загрузка изображения
    image = cv2.imread(image_path)
    # Преобразование изображения в оттенки серого
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Изменение размера изображения
    resized_image = cv2.resize(gray_image, target_size)
    # Инвертирование цветов
    inverted_image = cv2.bitwise_not(resized_image)
    # Приведение к типу данных с плавающей точкой
    normalized_image = inverted_image.astype(np.float32) / 255.0
    return normalized_image

predict_labels = [3, 7, 8]
predict_images = [torch.tensor(preprocess_image(f"./images/{label}.jpg")) for label in predict_labels]
predict_labels = torch.tensor(predict_labels)

model.eval()
all_preds = []
with torch.no_grad():
 for X in predict_images:
  preds = model(X)
  _, pred_labels = torch.max(preds, 1)
  print(torch.tensor(preds.numpy()))
  all_preds.extend(pred_labels.numpy())
visualize_predictions(predict_images, predict_labels.numpy(), all_preds, num_samples=3)