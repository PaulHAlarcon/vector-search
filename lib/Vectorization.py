
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.preprocessing import image
from numpy import expand_dims ,squeeze
from requests import post
import json


class MobileNetV2_features:    
    def  __init__(self) -> None:
        self.model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        self.collection_featurs=[]
        pass
    
    def collection_predic(self,root_dir):   #img_collection
        ''''
        var: root_dir  -> type string -
           - must be only a dir Path not byte , test or url  
        descrition: def collection_predic()
           - This function creates a collection of embedding vectors
        '''
        for dir_na_f in os.listdir(root_dir):
            self.collection_featurs.append(self.predict(os.path.join(root_dir,dir_na_f)))
    
    def preprocessing(self, source): #load_img
        img = image.load_img(source, target_size=(224, 224))
        img_array = expand_dims(image.img_to_array(img), axis=0)
        return preprocess_input(img_array)

    def predict(self, source):
        preprocessed_img = self.preprocessing(source)
        features = self.model.predict(preprocessed_img)
        features_vector = squeeze(features)
        return features_vector.flatten()


class Vision_Florence:
     
     def __init__(self,application=None) -> None:
        self.Vision_url = os.getenv('VISION_URL') +"/computervision"
        self.Vision_api = os.getenv('VISION_API')
        self.version = "?api-version=2023-02-01-preview&modelVersion=latest"
        self.vec_img_url = self.Vision_url + "/retrieval:vectorizeImage" + self.version  # For doing the image vectorization
        self.vec_txt_url = self.Vision_url + "/retrieval:vectorizeText" + self.version  # For the prompt vectorization
        self.application_list=['bytes','path','url','text']
        self.octet_stream_list=['bytes','path']
        self.collection_embedding=[]
        if application != None:
            self.set_application(application)
      
     def collection_predic(self,root_dir):   #img_collection
        ''''
        var: root_dir  -> type string -
           - must be only a dir Path not byte , test or url  
        descrition: def collection_predic()
           - This function creates a collection of embedding vectors
        '''
        for dir_na_eb in os.listdir(root_dir):
            self.collection_embedding.append(self.predict(os.path.join(root_dir,dir_na_eb),application='path'))

     def set_application(self,application : str):
        '''
        var: application  -> type string - 
           - 'path'  'application/octet-stream'  -> when you'd like use a bytes file
           - 'bytes'  'application/octet-stream'  -> when you'd like use a bytes file
           - 'url'    'application/json'          -> when you'd like use URL
           - 'text'   'application/json'          -> when you'd like use URL 
        descrition: def set_application()
           - This function sets API request headers, for resource input
        '''
        self.application=application
        self.headers = {
            'Ocp-Apim-Subscription-Key': self.Vision_api,
            'Content-type': 'application/json' if self.application not in self.octet_stream_list else 'application/octet-stream'  
        }
    
     def predict(self,source, application=None):
        '''
        var: source  -> type string or bytes 
           - must be a image bytes, url string or text  ,'path' ,'bytes', 'url', 'text'
        var: application -> type string
           - This function sets API request headers, for resource input
        descrition: def predict()
           - This function returns an embedding vector, can receive bytes or URLs of images
        '''
        if application =='path':
            with open(source ,'rb') as f:
                 source = f.read()
            application='bytes'

        if application != None:   
            self.set_application(application)

        response = post(
                  url=self.vec_img_url if application !='text' else self.vec_txt_url,
                  headers=self.headers, 
                  data= json.dumps({self.application: source}) if self.application not in self.octet_stream_list  else source)
        return response.json()['vector']

