import string
from random import choice
from tqdm import tqdm
import re

from lib.orm import Documents
from lib.vectorization import MobileNetV2_features, Vision_Florence
from lib.Analyze import ImageAnalyzer


def mobileNetV2_multi_feature(root_dir):
    model_feat=MobileNetV2_features()
    model_feat.collection_predic(root_dir=root_dir)
    return model_feat.collection_featurs

def florence_multi_embeddign(root_dir):
    flov=Vision_Florence()
    flov.collection_predic(root_dir=root_dir,type_='url')
    return flov.collection_embedding

def mobileNetV2_single_feature(source):
    model_feat=MobileNetV2_features()
    return model_feat.predict(source)

def florence_single_embedding_url(source,type_='url'):
    florv=Vision_Florence()
    florv.set_application('url')
    return florv.predict(source)

def florence_single_embedding_bytes(source):
    florv=Vision_Florence()
    florv.set_application('bytes')
    return florv.predict(source)

def built_documents(fiels,Vector_funtion) -> list :
    generate_random_id=lambda x : ''.join(choice(string.ascii_letters + string.digits) for _ in range(x))
    doc=Documents(id=generate_random_id,
                  URL=str,
                  Vector_img=Vector_funtion
                  )
    for img_n in tqdm(fiels):
        doc.add_document(id=12, URL=img_n, Vector_img=img_n)
    return doc.collection 

def built_documents_hybrid(fiels,Vector_funtion) -> list :
    generate_random_id=lambda x : ''.join(choice(string.ascii_letters + string.digits) for _ in range(x))
    analy=ImageAnalyzer()
    doc=Documents(id=generate_random_id,
                  URL=str,
                  Vector_img=Vector_funtion,
                  captions=str,
                  tags=list,
                  obj=list
                  )
    for img_n in tqdm(fiels):
        captions,tags,obj=analy.Analyzed(img_n,threshold=0.7,source_file=False)
        doc.add_document(id=12, URL=img_n, Vector_img=img_n,captions=captions[0],tags=tags,obj=obj)
    return doc.collection 

def caption_filter(captions,tags,objs) -> str :
        captions_str = ' '.join(captions)
        captions_=' '.join([w for w in captions_str.split(' ') if len(w)>2 ])   
        if objs==['']:
            tags_=re.sub(r'[^\w\s?!.]', '', ' '.join(tags)+' ' +' '.join(objs)) #+' '+captions_ 
            if len(tags)>5:
                tags=tags[:5]
            tags_=re.sub(r'[^\w\s?!.]', '', ' '.join(tags)+'. ' +captions_)
        else:
            tags_=' '.join(objs) 
        dict_tag={}
        for w in tags_.split(' '):
            if w in dict_tag:
                dict_tag[w]+=1
            else:
              dict_tag[w]=1
        output= ' '.join([x for x in dict_tag if dict_tag[x]>=1])
        return output


if __name__=='__main__':
  # ------ load env variable ------# 
  from dotenv import load_dotenv
  load_dotenv()
  # ------ load env variable ------# 
  with open(r'Data\fashion\train\black\105184860.jpg','rb') as f:
       source=f.read()
  analy=ImageAnalyzer()
  #captions,tags,obj=analy.Analyzed("https://saopenai.blob.core.windows.net/imgsearchtest/black/105266909.jpg",threshold=0.7,source_file=False)
  #print(caption_filter(captions,tags,obj))
  vf=Vision_Florence()
  print(vf.predict('ciao',application='text'))