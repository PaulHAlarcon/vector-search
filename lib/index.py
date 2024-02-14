from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient 
from azure.search.documents.models import VectorizedQuery
from annoy import AnnoyIndex
from os import getenv ,listdir ,path
from  json import dump ,load
import numpy as np

class LocalAnnoyIndex:
     def __init__(self,function_embedding=None ,vector_size=62720,metric='euclidean') -> None:
          self.annoy_index = AnnoyIndex(vector_size, metric=metric )
          self.function_embedding=function_embedding
     
     def Create_AnnoyIndex(self ,root_dir : str, output_dir_index=None ,output_dir_josn=None) -> None:
          output_dir_index=getenv('path_AnnoyIndex') if output_dir_index == None else output_dir_index
          output_dir_josn=getenv('path_json') if output_dir_josn is None else output_dir_josn
          list_feat=self.function_embedding(root_dir)
          self.path = {}
          for i,name_img in enumerate(listdir(root_dir)):
              self.path[str(i)]=path.join(root_dir, name_img)
              self.annoy_index.add_item(i,list_feat[i])
          self.annoy_index.build(n_trees=10) 
          self.annoy_index.save(output_dir_index)
          with open(output_dir_josn, 'w') as json_file:
              dump(self.path, json_file)
        
     def load_index(self,path_index=None,json_file=None):
          path_index=getenv('path_AnnoyIndex') if path_index == None else path_index
          json_file=getenv('path_json') if json_file == None else json_file
          with open(json_file, 'r') as json_file:
               self.path =load(json_file)
          self.annoy_index.load(path_index)

     def Search_AnnoyIndex(self,path_img_,n=3):
          features1 = self.function_embedding(path_img_)
          similar_image_indices = self.annoy_index.get_nns_by_vector(features1, n=n)
          output = [self.path[str(x)] for x in similar_image_indices]
          return output 
     
class AzureSearchIndex:
    def __init__(self,index_name=None):
        if index_name is None:
            self.index_name=getenv('Search_index_name')
        else:
            self.index_name=index_name
        self.credential = AzureKeyCredential(getenv('Search_key'))
        self.endpoint = f"https://{ getenv('Search_service_name')}.search.windows.net/"
        self.client = SearchClient(endpoint=self.endpoint, index_name=self.index_name, credential=self.credential)
        self.index_client= SearchIndexClient(self.endpoint, self.credential)
        
    def upload_documents(self,documents):
        self.client.upload_documents(documents=documents)
    
    def delete_index(self,index_name=None):
        if index_name is not None:
            self.index_name =index_name
        self.index_client.delete_index(self.index_name)

    def create_index(self,index : object):
        result = self.index_client.create_or_update_index(index)
        print(f' {result.name} created')

    def vector_search(self,vectr_embeddign ,k_n=3):
        vector_query = VectorizedQuery(vector=vectr_embeddign, k_nearest_neighbors=k_n, fields="Vector_img" )
        results = self.client.search(  
            search_text=None,  
            vector_queries= [vector_query],
            select=[],
        )  
        return results
        
    def semantic_search(self,text_to_serach):
        return self.client.search(search_text=text_to_serach)
    
    def vector_hybrid_search(self,test,vectr_embeddign ,k_n=3,exhaustive=False):
        vector_query2 = VectorizedQuery(vector=vectr_embeddign, k_nearest_neighbors=k_n, fields="Vector_img",exhaustive=exhaustive)
  
        results = self.client.search( 
            search_text=test, 
            search_fields=['captions'] ,
            vector_queries= [vector_query2],
            top=k_n,
            select=['Vector_img','captions','URL','id'],
        )  
        return results