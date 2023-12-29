import os
import json
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import (
    VectorizedQuery,
    VectorFilterMode,    
)
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class Search:
    
    def __init__(self, index: str):
        # assign the Search variables for Azure Cogintive Search - use .env file and in the web app configure the application settings
        AZURE_SEARCH_ENDPOINT = os.environ.get("AZURE_SEARCH_ENDPOINT")
        AZURE_SEARCH_ADMIN_KEY = os.environ.get("AZURE_SEARCH_ADMIN_KEY")
        #AZURE_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX")
        credential_search = AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY)
        self.embed_model = os.environ.get("OPENAI_EMBED_MODEL")
        print(f"creating search client with - {index}")
        self.sc = SearchClient(endpoint=AZURE_SEARCH_ENDPOINT, index_name=index, credential=credential_search)
        
        self.client = OpenAI()
    
    def get_embedding(self, text, model):
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input = [text], model=model).data[0].embedding
    
    def search_hybrid(self, query: str) -> str:
        print(f"Starting Search - {query}")
        vector_query = VectorizedQuery(vector=self.get_embedding(query, self.embed_model), k_nearest_neighbors=5, fields="contentVector")
        print("Got embedding")
        results = []
        
        r = self.sc.search(  
            search_text=None,  
            vector_queries= [vector_query],
            vector_filter_mode=VectorFilterMode.PRE_FILTER,
            select=["category", "sourcefile", "content"],
        )
        print("Got result")
        for doc in r:
                results.append(f"[SOURCEFILE:  {doc['sourcefile']}]" + doc['content'])
        print("\n".join(results))
        return ("\n".join(results))