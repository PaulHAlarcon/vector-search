from streamlit_selectable_image_gallery import image_gallery
import streamlit as st
from io import BytesIO
import base64

from lib.index import  AzureSearchIndex
from lib.vectorization import Vision_Florence

#from PIL import Image
# ------ load env variable ------# 
from dotenv import load_dotenv
load_dotenv()
# ------ load env variable ------# 
import requests


if 'Florence' not in st.session_state:
    st.session_state['Florence'] = Vision_Florence()
    st.session_state['Florence'].set_application('bytes')

if 'AzureSearch' not in st.session_state:
    st.session_state['AzureSearch'] = AzureSearchIndex('pvectortest')

def Azure_Search_img(img_path, threshold=0.01):
    emb=st.session_state['Florence'].predict(img_path)
    results=st.session_state['AzureSearch'].vector_search(emb,k_n=20)
    return [x['URL'] for x in results if x['@search.score'] >threshold] 

features=[]
similar=False
with st.sidebar:
    st.header('Image to Image')
    uploaded_file = st.file_uploader("Carica un'immagine", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Leggi i bytes del file caricato
        file_content = uploaded_file.getvalue()
        st.image(uploaded_file, caption="Immagine caricata", use_column_width=True)
        features =Azure_Search_img(BytesIO(file_content).read(),threshold=0.80)
        similar=True
        st.write(features)
 
if similar:
  st.header('Immagini simili:')
else:
  st.header('Archivio immagini:')

if len(features)>0:
    lsit_img=features
    if len(lsit_img)<1:
        height=250
    else:
        height=150

if features != []:
    images = []
    for file in  lsit_img :
        encoded = base64.b64encode(BytesIO(requests.get(file).content).read()).decode()
        images.append(f"data:image/jpeg;base64,{encoded}")
    selected_index = image_gallery(images,height)



