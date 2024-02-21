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
    
    # Return an array of all descriptors in an Embedding
    def get_descriptors(self, embedding_id: str, embeddingSetIndex: int = 0):    
        embedding = self.get(embedding_id)
        descriptors = []
        embeddingSet = embedding['embeddingSets'][embeddingSetIndex]  
        for embedding in embeddingSet['embeddings']:
            descriptors.append(embedding['descriptor'])
        return descriptors
    

