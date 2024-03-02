import requests
from io import BytesIO

class EmbeddingService:
    service_url: str = ""

    def __init__(self, service_url: str = None):
        self.service_url = service_url            

    def get(self, embedding_id: str):
        api_url: str = self.service_url + "/embeddings/" + embedding_id
        response = requests.get(api_url)
        return response.json()
    
    def get_content(self, embedding_id: str):
        api_url: str = self.service_url + "/embeddings/" + embedding_id
        response = requests.get(api_url)
        return response.content

    def get_all(self):
        api_url: str = self.service_url + "/embeddings"
        response = requests.get(api_url)
        return response.json()
    
    # Return an array of all descriptors in an Embedding
    def get_descriptors(self, embedding_id: str, embeddingSetIndex: int = 0):    
        embedding = self.get(embedding_id)
        descriptors = []
        embeddingSet = embedding['embeddingSets'][embeddingSetIndex]  
        for embedding in embeddingSet['embeddings']:
            descriptors.append(embedding['descriptor'])
        return descriptors
    
    # Build embedding from a list of descriptors
    def build_embedding(self, descriptors: list):
        embedding = {
            "embeddingSets": [
                {
                    "name": "ifs",
                    "embeddings": []
                }
            ]
        }
        for descriptor in descriptors:
            # convert tensor to array
            descriptor = descriptor.tolist()
            embedding['embeddingSets'][0]['embeddings'].append({
                "descriptor": descriptor
            })
        return embedding
    
    def get_image_from_embedding_ifs(self, embedding):
        api_url: str = self.service_url + "/decoder/ifs?width=128&height=128&rangeSize=4"
        headers = {'Content-Type': 'application/json'}
        response = requests.request("POST", api_url, headers=headers, json=embedding)
        if response.status_code != 200:
            print("get_image_from_embedding_ifs: Error", response.status_code)
            return None
        return BytesIO(response.content)