from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

os.environ["LANGCHAIN_PROJECT"] = "Sequential Chain Example"

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model1 = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.7, max_output_tokens=1000)
model2 = ChatGoogleGenerativeAI(model="gemini-3.8-flash",temperature=0.5, max_output_tokens=1000)

parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser

config = {
    'tags' : ['sequential_chain', 'example'],
    'metadata' : {'model': 'gemini-2.5-flash', 'version': '1.0','temperature': 0.7, 'max_output_tokens': 1000}
}

result1 = chain.invoke({'topic': 'Unemployment in India'}, config=config)

print(result1)
