import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# ----- config -----
EPOCHS = 1  # keep minimal for now
BATCH_SIZE = 64
LR = 0.001

# ----- data -----
transform = transforms.ToTensor()

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

# ----- model -----
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(32 * 32 * 3, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ----- train -----
for epoch in range(EPOCHS):
    for x, y in trainloader:
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

# ----- evaluate -----
correct = 0
total = 0

with torch.no_grad():
    for x, y in testloader:
        out = model(x)
        preds = torch.argmax(out, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

print("Baseline accuracy:", correct / total)
