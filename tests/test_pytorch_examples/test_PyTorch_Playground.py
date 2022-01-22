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
# # PyTorch playground

# + colab_type="code" id="PzCCniVwNTdp" colab={}
# Setting seeds to try and ensure we have the same results - this is not guaranteed across PyTorch releases.
import torch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import numpy as np

np.random.seed(0)

# + colab_type="code" id="fQLW-HL7_0pT" outputId="a2347d9a-bd38-4fa5-b11a-6c2ae5767134" colab={"base_uri": "https://localhost:8080/", "height": 34}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

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

# + colab_type="code" id="67eZUNEM5b7n" outputId="65d086ff-3a03-4cee-addd-315c55d7b626" colab={"base_uri": "https://localhost:8080/", "height": 102}
model.to(device)

# + [markdown] colab_type="text" id="XPdDu7KfWEfW"
# - The only change we have made to the code is that we are going to track the training loss, the testing loss and the accuracy across the 30 epochs.
# - We'll print out the train loss, the test loss and the accuracy after each epoch.
# - Because we are running this over 30 epochs this will take a bit longer to run - approx 15 minutes.

# + colab_type="code" id="VJLzWi0UqGWm" outputId="77ce05a3-3f82-4cef-cefc-8a43c4ddbafb" colab={"base_uri": "https://localhost:8080/", "height": 578}
from torch import optim

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 30
train_tracker, test_tracker, accuracy_tracker = [], [], []

for i in range(num_epochs):
    cum_loss = 0

    for batch, (images, labels) in enumerate(trainloader, 1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        cum_loss += loss.item()

    train_tracker.append(cum_loss / len(trainloader))
    print(f"Epoch({i + 1}/{num_epochs}) | Training loss: {cum_loss / len(trainloader)} | ", end='')

    test_loss = 0
    num_correct = 0
    total = 0

    for batch, (images, labels) in enumerate(testloader, 1):
        images = images.to(device)
        labels = labels.to(device)

        logps = model(images)
        batch_loss = criterion(logps, labels)
        test_loss += batch_loss.item()

        output = torch.exp(logps)
        pred = torch.argmax(output, 1)
        total += labels.size(0)
        num_correct += (pred == labels).sum().item()

    test_tracker.append(test_loss / len(testloader))
    print(f"Test loss: {test_loss / len(testloader)} | ", end='')
    accuracy_tracker.append(num_correct / total)
    print(f'Accuracy : {num_correct / total}')
print(f'\nNumber correct : {num_correct}, Total : {total}')
print(f'Accuracy of the model after 30 epochs on the 10000 test images: {num_correct * 100 / total}% ')

# + [markdown] colab_type="text" id="IqNpkYO6V9YI"
# - Has the accuracy of the model increased?
# - Now plot the training loss vs the test loss over 30 epochs.

# + colab_type="code" id="89a8FdTi-cNM" outputId="82701451-7413-4c00-d4e7-24f0e375a4c2" colab={"base_uri": "https://localhost:8080/", "height": 286}
import matplotlib.pyplot as plt

# %matplotlib inline
plt.plot(train_tracker, label='Training loss')
plt.plot(test_tracker, label='Test loss')
plt.legend()

# + [markdown] id="DHtKTHKKjG3r" colab_type="text"
# - Now add the accuracy to the mix.

# + colab_type="code" id="AJgyMHm2Pvx5" outputId="881d9ce0-19cd-4991-86a1-2d014ee00735" colab={"base_uri": "https://localhost:8080/", "height": 286}
import matplotlib.pyplot as plt

# %matplotlib inline
plt.plot(train_tracker, label='Training loss')
plt.plot(test_tracker, label='Test loss')
plt.plot(accuracy_tracker, label='Test accuracy')
plt.legend()

# + [markdown] id="APhVglVTk-og" colab_type="text"
# ## Further challenges and experiments
# - Can you get better accuracy from a model if you :
#     - Add more layers?
#     - Change the number of nodes in the layers?
#     - Train over fewer/higher epochs?
#     
# - Can you improve on your results if you add additional layers like [Dropout](https://pytorch.org/docs/master/nn.html#torch.nn.Dropout)

# +


# +


# +


# +


# +


# +


# +


# +
