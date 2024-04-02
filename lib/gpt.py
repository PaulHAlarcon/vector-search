from langchain_openai import AzureChatOpenAI
#from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
#from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser
#import json
from langchain_core.utils.function_calling import convert_to_openai_tool

class gpt_function_call:
    def __init__(self,prompt: object ,function: object, model = 'gpt35turbo16k') -> None:
        self.prompt = prompt
        self.query_function = [convert_to_openai_tool(function)['function']]
        self.function = function
        self.llm = AzureChatOpenAI(model=model,temperature=0)
        pass

    def function_invoke(self, image_description, user_text ):
        chain = (
                self.prompt
                | self.llm.bind(function_call={"name": self.query_function[0]['name']}, functions=self.query_function)
                | JsonOutputFunctionsParser()
                | self.function 
                )
        return chain.invoke({'image_description' : image_description, 'user_text' : user_text })

