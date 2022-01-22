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
#     language: python
#     name: python3
# ---

# + [markdown] colab_type="text" id="gXmCHcwKs6rd"
# # CPU to GPU

# + colab={} colab_type="code" id="PzCCniVwNTdp"
# Setting seeds to try and ensure we have the same results - this is not guaranteed across PyTorch releases.
import torch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import numpy as np

np.random.seed(0)
# -

torch.cuda.is_available()

# + colab={} colab_type="code" id="5S4Dfhtg5LyT"
device = torch.device('cuda')
# -

print(device)

# + colab={} colab_type="code" id="PCJzXv0OK1Bs"
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


# + colab={} colab_type="code" id="rqMqFbIVrbFH"
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
# -

model.to(device)

# + colab={} colab_type="code" id="VJLzWi0UqGWm"
from torch import optim

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 3

for i in range(num_epochs):
    cum_loss = 0

    for images, labels in trainloader:
        images.to(device)
        labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        cum_loss += loss.item()

    print(f"Training loss: {cum_loss / len(trainloader)}")

# + colab={} colab_type="code" id="UreWrEgjqQe3"
import matplotlib.pyplot as plt

# %matplotlib inline

images, labels = next(iter(testloader))

test_image_id = 0
img = images[test_image_id].view(1, 784)

with torch.no_grad():
    logps = model(img)

# + colab={} colab_type="code" id="2tRjoEDSqY8X"
ps = torch.exp(logps)
ps

# + colab={} colab_type="code" id="dpP_RLV-qkc6"
nps = ps.numpy()[0]
nps

# + colab={} colab_type="code" id="kBf23XrtqrB6"
FMNIST_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sport Shoes', 'Bag',
                 'Ankle Boot']
plt.xticks(np.arange(10), labels=FMNIST_labels, rotation='vertical')
plt.bar(np.arange(10), nps)


# + colab={} colab_type="code" id="2dwV20YRMwq5"
def denormalize(tensor):
    tensor = tensor * 0.5 + 0.5
    return tensor


img = img.view(28, -1)
img = denormalize(img)
plt.imshow(img, cmap='gray')

# + colab={} colab_type="code" id="AJgyMHm2Pvx5"
