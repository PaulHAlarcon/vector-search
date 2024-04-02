import streamlit as st 
from lib.index import  AzureSearchIndex ,LocalAnnoyIndex
from lib.Vectorization import Vision_Florence
from lib.Analyze import ImageAnalyzer
from io import BytesIO
from dotenv import load_dotenv
from lib.gpt import gpt_function_call
from st_clickable_images import clickable_images
from langchain_core.prompts import ChatPromptTemplate


def search_keys(search_text: str) :
    """this function receives as an argument a search text.

    Args:
        search_text: this function argument is for text search, it must contain a description of the item to be searched for
    """
    return search_text['search_text']

prompt = ChatPromptTemplate.from_messages([
    ("system", "you will get the descriptions of an image and the text of the user, you have to combine the two information in a coherent way."),
    ("user", "image_description: a yellow pants on a white background,\nuser_text: I need a neck warmer in this color"),
    ("ai", "Yellow-colored neck warmer"),
    ("user", "image_description: a red pants on a white background,\nuser_text: would like them to be short"),
    ("ai", "red shorts"),
    ("user", "image_description: {image_description} ,\nuser_text: {user_text}")
])

def main():
    load_dotenv()
    if 'Florence' not in st.session_state:
        st.session_state['Florence'] = Vision_Florence()

    if 'AzureSearch' not in st.session_state:
        st.session_state['AzureSearch'] = AzureSearchIndex('pvectortest')

    if 'ImageAnalyzer' not in st.session_state:
        st.session_state['ImageAnalyzer'] = ImageAnalyzer()

    if 'gpt_function_call' not in st.session_state:
        st.session_state['gpt_function_call'] = gpt_function_call(prompt=prompt, function=search_keys, model='gpt35_16k')

    if 'url_img_list' not in st.session_state:
        st.session_state['url_img_list'] =[]

    if 'clicked' not in st.session_state:
        st.session_state['clicked'] =-1


    def search(emb,k=1,test=None):
        '''
        -type_ : 'text' or 'bytes'
        '''
        result=st.session_state['AzureSearch'].vector_hybrid_search(test=test,
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
        keywords = st.text_input('Query Text',value='')#

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
                   st.image('https://static.vecteezy.com/system/resources/previews/023/506/653/original/photo-upload-icon-editable-vector.jpg', caption=" ", use_column_width=True)
        else:
                #st.empty()
                st.info('You can enhance your search by uploading an image.')

        #--------------------------Send Button 
        button_check = st.button("Send")
        #-------------- ------------Send Button 

    ##--------------------------Task
    if button_check : # Check condition Send Button
         try:
            with st.status(" Search images..."):
                st.write("Downloading data...")
                if loadimg and keywords != '':
                   text_emb=st.session_state['Florence'].predict(keywords,application='text')
                   img_emb=st.session_state['Florence'].predict(imge_bytes,application='bytes')
                   result=search(img_emb,k=1)
                   urels = [x['URL'] for i,x in enumerate(result) if x['@search.score']>=0]
                   captions,tags,objs=st.session_state['ImageAnalyzer'].Analyzed(sourse = urels[0],source_file=False)
                   text_key=st.session_state['gpt_function_call'].function_invoke(captions[0], keywords)
                   text_emb=st.session_state['Florence'].predict(text_key,application='text')
                   result=search(text_emb,k=12)
                   st.session_state['url_img_list'] = [x['URL'] for i,x in enumerate(result) if x['@search.score']>=0]

                elif not loadimg and keywords != '':
                   text_emb=st.session_state['Florence'].predict(keywords,application='text')
                   result=search(text_emb,k=12)
                   st.session_state['url_img_list']=[x['URL'] for i,x in enumerate(result) if x['@search.score']>=0]

                else:
                   img_emb=st.session_state['Florence'].predict(imge_bytes,application='bytes')
                   result=search(img_emb,k=12)
                   st.session_state['url_img_list']=[x['URL'] for i,x in enumerate(result) if x['@search.score']>=0.6]


         except:
            st.warning("An issue arose. Maybe there's a problem with the search text or image \n ")
            if loadimg:
               st.warning("If you don't insert an image, disable the Load Image flag")
            pass
    else:
        st.info('The results will be displayed in this space.')
        st.info('To get started you can search "shoes", "t-shirts" or "bags"')

    if st.session_state['url_img_list'] != []:
        with st.container(border=True):
             st.session_state['clicked'] = clickable_images(
                 st.session_state['url_img_list'],
                 titles=[f"Image #{str(i)}" for i in range(5)],
                 div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
                 img_style={"margin": "5px", "height": "200px"},
             )
        #st.markdown('<a target="_blank" href="https://www.globo.com/">Access globo.com</a>', unsafe_allow_html=True)




if __name__ == '__main__':
    from streamlit.web import cli as stcli
    from streamlit import runtime
    import sys
    if runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())