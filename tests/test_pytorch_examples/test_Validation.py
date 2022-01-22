# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + [markdown] colab_type="text" id="gXmCHcwKs6rd"
# # Validation

# + colab_type="code" id="PzCCniVwNTdp" colab={}
# Setting seeds to try and ensure we have the same results - this is not guaranteed across PyTorch releases.
import torch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import numpy as np

np.random.seed(0)

# + colab_type="code" id="PCJzXv0OK1Bs" colab={}
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch import nn

mean, std = (0.5,), (0.5,)

# Create a transform and normalise data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean, std)
                                ])

# Download FMNIST training dataset and load training data
trainset = datasets.FashionMNIST('~/.pytorch/FMNIST/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download FMNIST test dataset and load test data
testset = datasets.FashionMNIST('~/.pytorch/FMNIST/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


# + colab_type="code" id="rqMqFbIVrbFH" colab={}
class FMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)

        return x


model = FMNIST()

# + colab_type="code" id="oNNyI5YRZ7H1" colab={}
from torch import optim

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 3

for i in range(num_epochs):
    cum_loss = 0

    for images, labels in trainloader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        cum_loss += loss.item()

    print(f"Training loss: {cum_loss / len(trainloader)}")

# + colab_type="code" id="UWYw7ZOzsS8U" colab={}
import matplotlib.pyplot as plt

# %matplotlib inline

images, labels = next(iter(testloader))

test_image_id = 0
img = images[test_image_id].view(1, 784)

with torch.no_grad():
    logps = model(img)

# +
ps = torch.exp(logps)
ps

# +
nps = ps.numpy()[0]
nps

# +
FMNIST_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sport Shoes', 'Bag',
                 'Ankle Boots']
plt.xticks(np.arange(10), labels=FMNIST_labels, rotation='vertical')
plt.bar(np.arange(10), nps)


# +
def denormalize(tensor):
    tensor = tensor * 0.5 + 0.5
    return tensor


img = img.view(28, -1)
img = denormalize(img)
plt.imshow(img, cmap='gray')

# +
with torch.no_grad():
    num_correct = 0
    total = 0

    cnt = 0
    for images, labels in testloader:

        logps = model(images)
        output = torch.exp(logps)
        print(output)
        cnt += 1

        if cnt > 0:
            break

# +
with torch.no_grad():
    num_correct = 0
    total = 0

    # set_trace()
    for images, labels in testloader:
        logps = model(images)
        output = torch.exp(logps)

        pred = torch.argmax(output, 1)
        total += labels.size(0)

# +
pred, labels

# +
pred == labels

# + colab_type="code" id="6V-3r9n-iCMb" colab={}
with torch.no_grad():
    num_correct = 0
    total = 0

    # set_trace()
    for images, labels in testloader:
        logps = model(images)
        output = torch.exp(logps)

        pred = torch.argmax(output, 1)
        total += labels.size(0)
        num_correct += (pred == labels).sum().item()

    print(f'Accuracy of the model on the 10000 test images: {num_correct * 100 / total}% ')

# +
