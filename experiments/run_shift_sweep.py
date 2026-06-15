import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import copy

from shiftlab.shifts.noise_shift import apply_noise_shift

# ----- config -----
sigmas = [0.0, 0.1, 0.2, 0.3, 0.4]
BATCH_SIZE = 64
EPOCHS = 1
LR = 0.001

transform = transforms.ToTensor()

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

results = []

for sigma in sigmas:

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(32 * 32 * 3, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )

    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    # ----- train -----
    for epoch in range(EPOCHS):
        for x, y in trainloader:
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

    # ----- evaluate under shift -----
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE):
            x_shifted = apply_noise_shift(x, sigma)
            out = model(x_shifted)
            preds = torch.argmax(out, dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total
    results.append((sigma, acc))

    print(f"sigma={sigma} accuracy={acc}")

np.save("shift_results.npy", results)
