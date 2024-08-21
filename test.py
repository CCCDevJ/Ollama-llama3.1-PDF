# from langchain_community.llms import Ollama
#
# llm = Ollama(model="llama3.1")
#
# response = llm.invoke("Tell me a cate joke.")
#
# print(response)

import langchain_community
import flask
import langchain
import pydantic
import fastapi

print("langchain==", langchain.__version__)
print("langchain_community==", langchain_community.__version__)
print("pydantic==", pydantic.__version__)
print("fastapi==", fastapi.__version__)
print("langchain_community==", langchain_community.__version__)
