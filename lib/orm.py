from azure.storage.blob import BlobServiceClient
from os import makedirs,path,getenv
from io import BytesIO

class Documents:
    def __init__(self, **kwargs) -> None:
        self.collection = []
        self.json_data = kwargs

    def add_document(self,**kwargs ):
       self.collection.append({var:self.json_data[var](kwargs[var]) for var in kwargs})


class AzureBlobFiles:
    def __init__(self,MY_BLOB_CONTAINER):
        print("Intializing AzureBlobFiles")      
        self.blob_service_client = BlobServiceClient.from_connection_string( getenv('MY_CONNECTION_STRING_blob'))
        self.MY_BLOB_CONTAINER=MY_BLOB_CONTAINER
        self.my_container = self.blob_service_client.get_container_client(MY_BLOB_CONTAINER)
        
    def save_blob(self, file_name, file_content,LOCAL_BLOB_PATH,BLOBNAME):
        download_file_path =  path.join(LOCAL_BLOB_PATH, file_name)
        makedirs( path.dirname(download_file_path), exist_ok=True)
        with open(download_file_path, "wb") as file:
            file.write(file_content)

    def download_all_blobs_in_container(self):
        my_blobs = self.my_container.list_blobs()
        for blob in my_blobs:
            bytes = self.my_container.get_blob_client(blob).download_blob().readall()
            self.save_blob(blob.name, bytes) 

    def download_all_blobs_in_Stream(self ,parfi='o'):
        my_blobs = self.my_container.list_blobs()
        for blob in my_blobs:
            name= path.basename( path.normpath(blob.name))
            if  name.split('_')[1]=='1' or name.split('_')[1]==parfi:
                   stream = BytesIO()
                   self.my_container.get_blob_client(blob).download_blob().readinto(stream)
                   stream.seek(0)
                   yield stream,blob.name

    def get_files_urls(self):
        blob_list = self.my_container.list_blobs()
        return [f"{self.blob_service_client.url}{self.MY_BLOB_CONTAINER}/{blob.name}" for blob in blob_list]