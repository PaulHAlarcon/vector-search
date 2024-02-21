import os
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader ,ReadTheDocsLoader ,TextLoader
from langchain_openai import  AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from operator import itemgetter
from langchain_core.utils.function_calling import convert_to_openai_tool


def multiply(a: int, b: int) -> int:
    """Multiply two integers together.

    Args:
        a: First integer
        b: Second integer
    """
    return a * b

def Image_search(description: str) -> int:
    """Search for an image from a description

    Args:
        description: decsrizion of image
    """
    return description

class retriever():
    def __init__(self,embeddings) -> None:
         self.documents=[]
         self.embeddings=embeddings
    
    def add_document(self,url):
        documents = WebBaseLoader(url).load()
        documents = RecursiveCharacterTextSplitter().split_documents(documents)
        for doc in documents:
            self.documents.append(doc)

    def set_FAISS(self):
        self.vector = FAISS.from_documents(self.documents, self.embeddings)
        return self.vector.as_retriever()

def chat_chain(llm,retriever):
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context},  You can update the context if the user adds a new web resource"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
    return retrieval_chain

def chat_pipe_chain(llm,retriever): 
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}, You are a fashion assistant who is able to search for product images, and you have to help customers search for their product better. To talk to the user try to understand his tastes and his visdreams , style , which occazine want to dress for, man , woman , name. ect.. "),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    chain = (
    {
        "context": itemgetter("input") | retriever,
        "chat_history": itemgetter("chat_history"),
        "input":  itemgetter("input"),
    }
    | prompt
    | llm
     )
    return chain

if __name__=='__main__':
    # ------ load env variable ------# 
    from dotenv import load_dotenv
    load_dotenv()
    # ------ load env variable ------# 
    llm = AzureChatOpenAI(model='gpt35_16k')#os.getenv('deployment_name')
    llm = llm.bind(tools=[convert_to_openai_tool(Image_search)])
    embeddings = AzureOpenAIEmbeddings(model='ada')
    chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
    list_url=['https://ricette.giallozafferano.it/Pasta-al-forno.html']
    rtv=retriever(embeddings)
    rtv.add_document(list_url)
    rtv_faiss=rtv.set_FAISS()
    bot=chat_pipe_chain(llm,rtv_faiss)
    print(bot.invoke({'input':"ciao come stai?, vorrei un immagine di scarpe rose",'chat_history':chat_history}))
    