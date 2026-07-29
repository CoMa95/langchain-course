from dotenv import load_dotenv
import os

from langsmith import traceable
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()

@traceable
def main():
    print("Hello from langchain-course!")
    information = """
    Born into the wealthy Musk family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he has Canadian citizenship since his mother was born there. He received bachelor's degrees in 1997 from the University of Pennsylvania before moving to California to pursue business ventures. In 1995, Musk co-founded Zip2, a web software company. Following its sale in 1999, he co-founded X.com, an e-commerce payment system that merged with Confinity in March 2000 to form PayPal, which was acquired by eBay in 2002. Musk also became an American citizen in 2002.
    """

    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. Two (if possible) interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables= ['information'], template = summary_template
    )

    llm = ChatOpenAI(temperature = 0, model = 'gpt-5', api_key=os.getenv('OPEN_API_KEY'))
    #llm = ChatOllama(temperature = 0, model = 'gemma3:1b')
    chain = summary_prompt_template | llm # LangChain Expression Language operator
    response = chain.invoke(input={'information': information})

    print(response.content)


if __name__ == "__main__":
    main()
