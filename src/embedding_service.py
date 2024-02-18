import requests

class EmbeddingService:
    service_url: str = ""

    def __init__(self, service_url: str = None):
        self.service_url = service_url            

    def get(self, embedding_id: str):
        api_url: str = self.service_url + "/embeddings/" + embedding_id
        response = requests.get(api_url)
        return response.json()
    
    def get_all(self):
        api_url: str = self.service_url + "/embeddings"
        response = requests.get(api_url)
        return response.json()
    
