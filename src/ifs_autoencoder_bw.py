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

import json
from PIL import Image

from embedding_service import EmbeddingService

##
hardware='cpu'

#
imageWidth = 128
imageHeight = 128
descriptors_length = 6
rangeNumber = int((imageWidth / 4) * (imageHeight / 4))

##
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(6144, 2304),
            nn.Tanh(),
            nn.Linear(2304, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 8),
            nn.Tanh(),
            nn.Linear(8, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 1024),
            nn.Tanh(),
            nn.Linear(1024, 2304),
            nn.Tanh(),
            nn.Linear(2304, 6144),
            nn.Sigmoid()            
        )
    
    def forward(self, inp):
        # reshape
        inp = inp.view(-1, rangeNumber * descriptors_length)
        latent = self.encoder(inp)
        
        # See https://wandb.ai/ayush-thakur/dl-question-bank/reports/An-Introduction-To-The-PyTorch-View-Function--VmlldzoyMDM0Nzg
        out = self.decoder(latent).view(-1, rangeNumber, descriptors_length)
        return out, latent
    

service_url: str = "http://127.0.0.1:23175"
embeddingService = EmbeddingService(service_url)

# Get all embeddings
#embeddings = embeddingService.get_all()

# Display the latent vector for one image
# 65e3364a948eeb762a82cd66 is the small girl
#embedding_sample_id = "65e3364a948eeb762a82cd66"
#embedding_sample = embeddingService.get_descriptors(embedding_sample_id)
embedding_sample = embeddingService.get_descriptors_from_file('data/test/embedding_sample.json')
print("embedding_sample size", len(embedding_sample))

# For testing
#image_data = embeddingService.get_image_from_embedding_ifs(embeddingService.get_content(embedding_sample_id))
#image = Image.open(image_data)
#image.show()

inputs = torch.tensor(embedding_sample, dtype=torch.float32)
print("input shape", inputs.shape)
print("input", inputs)
inputs = inputs.to(hardware)

##
autoencoder = AutoEncoder().to(hardware)

first_parameter = next(autoencoder.parameters())
input_shape = first_parameter.size()
#print("first parameter", first_parameter)   
#print("input shape", input_shape)

# Encode the image
output, latent = autoencoder(inputs)
print("output size", output.shape)
print("output", output)

## decode
output_embedding_descriptors = autoencoder.decoder(latent)
output_embedding_descriptors = torch.reshape(output_embedding_descriptors, (rangeNumber, descriptors_length))
print("output_embedding shape", output_embedding_descriptors.shape)
print("output_embedding", output_embedding_descriptors)

output_embedding = embeddingService.build_embedding(output_embedding_descriptors)
#print("output_embedding", output_embedding)
with open('embedding.json', 'w') as f:
  json.dump(output_embedding, f, ensure_ascii=False)

##
num_epochs = 1
loss_fn = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

#
#embeddings = embeddingService.get_all()
#embeddings = [embedding_sample_id]

#training loop
for epoch in range(num_epochs):
    #for id in embeddings:
    print("Trainig with id:", id)
    #descriptors = embeddingService.get_descriptors(id)
    descriptors = embeddingService.get_descriptors_from_file('data/test/embedding_sample.json')
    inputs = torch.tensor(descriptors, dtype=torch.float32)
    inputs = inputs.to(hardware)
    optimizer.zero_grad()
    outputs, _ = autoencoder(inputs)
    outputs = torch.reshape(outputs, (rangeNumber, descriptors_length))
    loss = loss_fn(outputs, inputs)
    loss.backward()
    optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

## Generate from the latent space
import numpy as np

num_latent = 6
latent_x = np.linspace(-1, 1, num=num_latent)
latent_y = np.linspace(-1, 1, num=num_latent)

X, Y = np.meshgrid(latent_x, latent_y)
latents = np.stack((X, Y), axis=-1)
latents = torch.FloatTensor(latents).to(hardware)

autoencoder.eval()
output_embedding_descriptors_array = autoencoder.decoder(latents)
print("output_embedding_descriptors_array shape", output_embedding_descriptors_array.shape)
fig, axs = plt.subplots(num_latent, num_latent, figsize=(6, 6))

for i in range(num_latent):
    for j in range(num_latent):        
        ax = axs[i, j]
        output_embedding_desc = output_embedding_descriptors_array[i, j]
        output_embedding_desc = output_embedding_descriptors_array[i, j].view(-1, rangeNumber, descriptors_length).detach().cpu()
        output_embedding_desc = torch.reshape(output_embedding_desc, (rangeNumber, descriptors_length))
        print("output_embedding_desc shape", output_embedding_desc.shape)
        print("output_embedding_desc", output_embedding_desc)

        # Decode the image
        output_embedding = embeddingService.build_embedding(output_embedding_desc)
        with open('embedding1.json', 'w') as f:
            json.dump(output_embedding, f, ensure_ascii=False)
        image_data = embeddingService.get_image_from_embedding_ifs(output_embedding)
        image = Image.open(image_data)

        # Display the image
        ax.set_title('{:.1f},{:.1f}'.format(latent_x[i], latent_y[j]), fontsize=8)
        ax.imshow(image, cmap='gray')
        ax.axis('off')    
plt.savefig(f'./result_latent.jpg')    
plt.show()
