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
            nn.Linear(73728, 18432),
            nn.Tanh(),
            nn.Linear(18432, 2304),
            nn.Tanh(),
            nn.Linear(2304, 256),
            nn.Tanh(),
            nn.Linear(256, 18),
            nn.Tanh(),
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
            nn.Linear(18, 256),
            nn.Tanh(),
            nn.Linear(256, 2304),
            nn.Tanh(),
            nn.Linear(2304, 18432),
            nn.Tanh(),
            nn.Linear(18432, 73728),
            nn.Sigmoid()
        )
    
    def forward(self, inp):
        # reshape 18 * 4096 image into 73728
        inp = inp.view(-1, 4096 * 18)
        print("inp", inp)
        latent = self.encoder(inp)
        
        # See https://wandb.ai/ayush-thakur/dl-question-bank/reports/An-Introduction-To-The-PyTorch-View-Function--VmlldzoyMDM0Nzg
        out = self.decoder(latent).view(-1, 4096, 18)
        return out, latent
    

service_url: str = "http://127.0.0.1:16217"
embeddingService = EmbeddingService(service_url)

# Get all embeddings
embeddings = embeddingService.get_all()

# Display the latent vector for one image
get_descriptors = embeddingService.get_descriptors("65bfaf0c7371e83a29801a08")
print("get_descriptors size", len(get_descriptors))

inputs = torch.tensor(get_descriptors, dtype=torch.float32)
print("input shape", inputs.shape)
inputs = inputs.to(hardware)

##
autoencoder = AutoEncoder().to(hardware)

first_parameter = next(autoencoder.parameters())
input_shape = first_parameter.size()
#print("first parameter", first_parameter)   
#print("input shape", input_shape)

output, latent = autoencoder(inputs)
print("output size", output.shape)
print("output", output)

##
num_epochs = 10
loss_fn = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

#training loop
for epoch in range(num_epochs):
    for id in embeddings:
        print("Trainig with id:", id)
        descriptors = embeddingService.get_descriptors(id)
        print("get_descriptors size", len(descriptors))
        inputs = torch.tensor(descriptors, dtype=torch.float32)
        inputs = inputs.to(hardware)
        optimizer.zero_grad()
        outputs, _ = autoencoder(inputs)
        outputs = torch.reshape(outputs, (4096, 18))
        print("inputs shape", inputs.shape)
        print("outputs shape", outputs.shape)
        loss = loss_fn(outputs, inputs)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
