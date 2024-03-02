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


##
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
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
            nn.Sigmoid()
        )
    
    def forward(self, inp):
        # reshape 18 * 1024 image into 73728 digits
        inp = inp.view(-1, 1024 * 18)
        latent = self.encoder(inp)
        
        # See https://wandb.ai/ayush-thakur/dl-question-bank/reports/An-Introduction-To-The-PyTorch-View-Function--VmlldzoyMDM0Nzg
        out = self.decoder(latent).view(-1, 1024, 18)
        return out, latent
    

service_url: str = "http://127.0.0.1:23175"
embeddingService = EmbeddingService(service_url)

# Get all embeddings
#embeddings = embeddingService.get_all()

# Display the latent vector for one image
# 65e3364a948eeb762a82cd66 is the small girl
embedding_sample_id = "65e3364a948eeb762a82cd66"
embedding_sample = embeddingService.get_descriptors(embedding_sample_id)
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
output_embedding_descriptors = torch.reshape(output_embedding_descriptors, (1024, 18))
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
embeddings = [embedding_sample_id]

#training loop
for epoch in range(num_epochs):
    for id in embeddings:
        print("Trainig with id:", id)
        descriptors = embeddingService.get_descriptors(id)
        inputs = torch.tensor(descriptors, dtype=torch.float32)
        inputs = inputs.to(hardware)
        optimizer.zero_grad()
        outputs, _ = autoencoder(inputs)
        outputs = torch.reshape(outputs, (1024, 18))
        loss = loss_fn(outputs, inputs)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

##
axis = []
labels = []
with torch.no_grad():
    #for i, (img, label) in enumerate(train_loader):
    for id in embeddings:
        embedding = embeddingService.get_descriptors(id)
        inputs = torch.tensor(embedding, dtype=torch.float32)
        inputs = inputs.to(hardware)
        _, latent = autoencoder(inputs) 
        latent = latent.detach().cpu()
        axis.extend(latent)
        labels.extend([1])
        
fig, ax = plt.subplots(figsize=(8, 8))
for i in range(len(labels)):
    ax.text(axis[i][0], axis[i][1], str(labels[i]), 
            color=plt.cm.Set1(labels[i]))
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.savefig(f'./ifs_result_img.jpg')
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
output_embedding_descriptors_array = autoencoder.decoder(latents)
print("output_embedding_descriptors_array shape", output_embedding_descriptors_array.shape)
fig, axs = plt.subplots(num_latent, num_latent, figsize=(6, 6))

for i in range(num_latent):
    for j in range(num_latent):        
        ax = axs[i, j]
        output_embedding_desc = output_embedding_descriptors_array[i, j]
        output_embedding_desc = output_embedding_descriptors_array[i, j].view(-1, 1024, 18).detach().cpu()
        output_embedding_desc = torch.reshape(output_embedding_desc, (1024, 18))
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
