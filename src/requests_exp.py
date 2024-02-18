import requests
import os

from embedding_service import EmbeddingService

service_url: str = "http://127.0.0.1:58359"

embeddingService = EmbeddingService(service_url)

embedding = embeddingService.get("65bfaf0c7371e83a29801a08")
print(embedding['id'])
print(embedding['name'])
print(embedding['embeddingSets'][0]['embeddings'][0]['descriptor'])

print("All embedding ids")
embeddings = embeddingService.get_all()
print(embeddings)

# Print all the embeddings
print("All embeddings")
for id in embeddings:
    embedding = embeddingService.get(id)
    print(embedding['embeddingSets'][0]['embeddings'][0]['descriptor'])

#
api_url = "http://127.0.0.1:58359/analyzer/ifs"

print("Sending request to ", api_url)

## Send a POST request with a binary file payload
with open("data/test/image1.jpg", "rb") as file:
     requests.post(api_url, data=file)

## Send a POST request sending a binary file with a specific key and multipart/form-data header
path_img = "data/test/laptop.png"
payload = {}
headers = {}
with open(path_img, "rb") as img:
    name_img = os.path.basename(path_img)
    files = {'file': (name_img, img)}
    response = requests.request("POST", api_url, headers=headers, data=payload, files=files)
    embedding = response.json()
    print(embedding['id'])
    print(embedding['name'])
    print(embedding['embeddingSets'][0]['embeddings'][0]['descriptor'])

