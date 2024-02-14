import streamlit as st
#from streamlit_selectable_image_gallery import image_gallery
from lib.index import  AzureSearchIndex ,LocalAnnoyIndex
from lib.Vectorization import Vision_Florence
from lib.Analyze import ImageAnalyzer
from io import BytesIO
from dotenv import load_dotenv
import os 
import base64
import requests
from st_clickable_images import clickable_images

import time
# initialize float feature/capability



load_dotenv()
if 'Florence' not in st.session_state:
    st.session_state['Florence'] = Vision_Florence()

if 'AzureSearch' not in st.session_state:
    st.session_state['AzureSearch'] = AzureSearchIndex('pvectortest')

if 'url_img_list' not in st.session_state:
    st.session_state['url_img_list'] =[]

if 'clicked' not in st.session_state:
    st.session_state['clicked'] =-1





def search(emb,k=1):
    '''
    -type_ : 'text' or 'bytes'
    '''
    result=st.session_state['AzureSearch'].vector_hybrid_search(test=None,
                                                                     vectr_embeddign=emb,
                                                                     k_n=k,
                                                                     exhaustive=True)
    return result

def local_search(result_list,source,threshold=0.85,n=10):
    '''
    -type_ : 'text' or 'bytes'
    '''
    ris_text=LocalAnnoyIndex(vector_size=1024,function_embedding=st.session_state['Florence'].predict)
    Url={}
    Vector={}
    for i,x in enumerate(result_list):
        if x['@search.score']>=threshold:
           Url[str(i)]=x['URL']
           Vector[str(i)]=x['Vector_img']
           ris_text.annoy_index.add_item(i,x['Vector_img'])
        pass
    ris_text.path=Url
    ris_text.annoy_index.build(n_trees=10)
    similar_image_indices = ris_text.annoy_index.get_nns_by_vector(source, n=n)
    fil = [ris_text.path[str(x)] for x in similar_image_indices]
    return fil 
    
with st.sidebar:
    #------------------------Titole
    st.title("Image Retrieval")
    #------------------------Titole
    keywords = title = st.text_input('Query Text',value='')#
    
    loadimg = st.toggle('load Image')
    col =st.columns(2)
    
         
    if loadimg:
        with col[0]:
            uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        # To read file as bytes:
        if uploaded_file is not None:
            
            with col[1]:  
              st.image(uploaded_file, caption=" ", use_column_width=True)

            imge_bytes = uploaded_file.getvalue()
            imge_bytes= BytesIO(imge_bytes).read()
        else:
            imge_bytes=0
            with col[1]:
               st.image('Data\log_img.jpg', caption=" ", use_column_width=True)
    else:
            #st.empty()
            st.info('You can enhance your search by uploading an image.')
    #--------------------------Send Button 
    button_check = st.button("Send")
    #-------------- ------------Send Button 
import numpy as np
##--------------------------Task
if button_check : # Check condition Send Button
     try:
        with st.status(" Search images..."):
            #st.write("Search images...")
            st.write("Downloading data...")
            if loadimg and keywords != '':
               text_emb=st.session_state['Florence'].predict(keywords,application='text')
               img_emb=st.session_state['Florence'].predict(imge_bytes,application='bytes')
               emb_g=np.array(text_emb)+np.array(img_emb)
               result=search(emb_g,k=1)
               result=[r['Vector_img'] for r in result if r['@search.score']>0.7]
               if len(result)>0:
                  normal_emb=result[0]
                  imgresult=search(normal_emb,k=900)
                  st.write("Local search")
                  st.session_state['url_img_list']=local_search(imgresult,source=text_emb,threshold=0.8,n=12)
                  st.write("End search")
    
            elif not loadimg and keywords != '':
               text_emb=st.session_state['Florence'].predict(keywords,application='text')
               result=search(text_emb,k=12)
               st.session_state['url_img_list']=[x['URL'] for i,x in enumerate(result) if x['@search.score']>=0]
               
            else:
               img_emb=st.session_state['Florence'].predict(imge_bytes,application='bytes')
               result=search(img_emb,k=12)
               st.session_state['url_img_list']=[x['URL'] for i,x in enumerate(result) if x['@search.score']>=0.6]
            
            #with st.sidebar:
            #     st.write(url_img_list)
     except:
        st.warning('An issue arose.')
        pass
else:
    st.info('The results will be displayed in this space.')

if st.session_state['url_img_list'] != []:
    with st.container(border=True):
         st.session_state['clicked'] = clickable_images(
             st.session_state['url_img_list'],
             titles=[f"Image #{str(i)}" for i in range(5)],
             div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
             img_style={"margin": "5px", "height": "200px"},
         )
    #st.markdown('<a target="_blank" href="https://www.globo.com/">Access globo.com</a>', unsafe_allow_html=True)
