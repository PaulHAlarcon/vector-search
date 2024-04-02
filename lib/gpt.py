from langchain_openai import AzureChatOpenAI
#from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
#from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser
#import json
from langchain_core.utils.function_calling import convert_to_openai_tool

class gpt_function_call:
    def __init__(self,prompt: object ,function: object, model = 'gpt35turbo16k') -> None:
        '''
        var: prompt  -> type object (langchain: object ChatPromptTemplate)
           - ChatPromptTemplate prompt of langchain to handle input parameters
        var: function -> type object (function)
           - function call name
        var: model -> type string
           - gpt model name published
        '''
        self.prompt = prompt
        self.query_function = [convert_to_openai_tool(function)['function']]
        self.function = function
        self.llm = AzureChatOpenAI(model=model,temperature=0)
        self.function_chain(self)
        pass

    def function_chain(self):
        '''
        descrition: def function_chain()
           - this functione sets up the function call chain
        '''
        self.chain = (
                self.prompt
                | self.llm.bind(function_call={"name": self.query_function[0]['name']}, functions=self.query_function)
                | JsonOutputFunctionsParser()
                | self.function 
                )
    
    def invoke(self,arg):
        '''
        var: arg  -> type dict 
           - dictionary with the prompt parameters 
        descrition: def invoke()
           - this function returns the result of the function call chain
        '''
        return self.chain.invoke(arg)


