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

from embedding_service import EmbeddingService

##
hardware='cpu'


##
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(18, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 2),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 16),
            nn.Tanh(),
            nn.Linear(16, 18),
            nn.Tanh(),
        )
    
    def forward(self, inp):
        # reshape 28 * 28 image into 784
        inp = inp.view(-1, 18)
        latent = self.encoder(inp)
        
        out = self.decoder(latent).view(-1, 18)
        return out, latent
    
##
autoencoder = AutoEncoder().to(hardware)

##
num_epochs = 10
loss_fn = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

service_url: str = "http://127.0.0.1:58359"
embeddingService = EmbeddingService(service_url)

# Display the latent vector for one image
embedding = embeddingService.get("65bfaf0c7371e83a29801a08")
inputs = torch.tensor(embedding['embeddingSets'][0]['embeddings'][0]['descriptor'], dtype=torch.float32)
print("input", inputs)
inputs = inputs.to(hardware)
output, latent = autoencoder(inputs)
print("output", output)
print("latent", latent)
