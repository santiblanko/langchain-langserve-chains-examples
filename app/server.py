#!/usr/bin/env python
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langserve import add_routes
import os
from dotenv import load_dotenv
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.llms import OpenAI

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    ChatOpenAI(openai_api_key=OPENAI_API_KEY),
    path="/openai",
)

# Create the OpenAI model
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY,model_name='gpt-3.5-turbo', temperature=1)
#model = OpenAI(temperature=0.7, openai_api_key=OPENAI_API_KEY)

synopsis_prompt = PromptTemplate.from_template(
    """
 "Return a JSON object with `title` and `history` key that answers the following question: You are a historian tasked with providing information about a title provided: {title}. The sypnosis must have 50 characters."
"""
)


review_prompt = PromptTemplate.from_template(
    """
        You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.
        Play Synopsis: (use the history key)
        {synopsis}
        Review from a New York Times play critic of the above play in JSON format with title, history and synopsis key in 100 characters:
    """
)

translate_prompt = PromptTemplate.from_template(
    """ devuelve el objeto json con history (ingles), synopsis (ingles) y translate (español).
        {synopsis}

    """
)

review_chain = LLMChain(llm=llm, prompt=review_prompt)
translate_chain = LLMChain(llm=llm, prompt=translate_prompt)
json_parser = SimpleJsonOutputParser()

overall_chain = SimpleSequentialChain(
    chains=[review_chain, translate_chain],
    verbose=True
)

add_routes(
    app,
     synopsis_prompt | llm | overall_chain,
    path="/chain",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)