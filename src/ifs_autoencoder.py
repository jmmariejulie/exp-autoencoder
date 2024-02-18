#Also in the GoogleColab notebook for the full code: https://colab.research.google.com/drive/1QyWkCc7vLe6wpuC0Cb6jfxcBSE4u4d7h#scrollTo=LIEpaDjE2IAY

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch
from torchvision.transforms import transforms
import torchvision

from keras.datasets import mnist

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

##
hardware='cpu'


##
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 2),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )
    
    def forward(self, inp):
        # reshape 28 * 28 image into 784
        inp = inp.view(-1, 784)
        latent = self.encoder(inp)
        
        out = self.decoder(latent).view(-1, 1, 28, 28)
        return out, latent
    
##
autoencoder = AutoEncoder().to(hardware)

##
num_epochs = 10
loss_fn = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

#create dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Display the latent vector for one image
inputs, _ = train_loader.dataset[1]
inputs = inputs.to(hardware)
output, latent = autoencoder(inputs)
print(latent)

#training loop
for epoch in range(num_epochs):
    for data in train_loader:
        inputs, _ = data
        inputs = inputs.to(hardware)
        optimizer.zero_grad()
        outputs, _ = autoencoder(inputs)
        loss = loss_fn(outputs, inputs)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

##
axis = []
labels = []
with torch.no_grad():
    for i, (img, label) in enumerate(train_loader):
        _, latent = autoencoder(img.to(hardware)) 
        latent = latent.detach().cpu()
        axis.extend(latent)
        labels.extend(label.tolist())
        
fig, ax = plt.subplots(figsize=(8, 8))
for i in range(len(labels)):
    ax.text(axis[i][0], axis[i][1], str(labels[i]), 
            color=plt.cm.Set1(labels[i]))
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.savefig(f'./result_img.jpg')
plt.show()

## Generate from the latent space
import numpy as np

num_latent = 6
latent_x = np.linspace(-1, 1, num=num_latent)
latent_y = np.linspace(-1, 1, num=num_latent)

X, Y = np.meshgrid(latent_x, latent_y)
latents = np.stack((X, Y), axis=-1)
latents = torch.FloatTensor(latents).to(hardware)

autoencoder.eval()
imgs = autoencoder.decoder(latents)
fig, axs = plt.subplots(num_latent, num_latent, figsize=(6, 6))

for i in range(num_latent):
    for j in range(num_latent):
        ax = axs[i, j]
        img = imgs[i, j].view(28, 28, 1).detach().cpu()

        # Display the image
        ax.set_title('{:.1f},{:.1f}'.format(latent_x[i], latent_y[j]), fontsize=8)
        ax.imshow(img, cmap='gray')
        ax.axis('off')    
plt.savefig(f'./result_latent.jpg')    
plt.show()
