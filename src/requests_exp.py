import requests
import os

api_url = "http://127.0.0.1:11682/embeddings/65bfaf0c7371e83a29801a08"
response = requests.get(api_url)
print(response.json())

embedding = response.json()
print(embedding['id'])
print(embedding['name'])
print(embedding['embeddingSets'][0]['embeddings'][0]['descriptor'])

api_url = "http://127.0.0.1:11682/analyzer/ifs"

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
    print(response.text)