from dotenv import load_dotenv
import os

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain_openai import ChatOpenAI

# Option A - manual tool creation with the Tavily SDK
# from tavily import TavilyClient
# tavily = TavilyClient()
# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches over Internet
#     Args:
#         query: the query to search for
#     Returns:
#         Search result
#     """
#     print(f'Searching for "{query}"')
#     return tavily.search(query=query)
# a
# Option B - official tool from the Tavily-LangChain integration
from langchain_tavily import TavilySearch

llm = ChatOpenAI(api_key=os.getenv('OPEN_API_KEY'))
#tools = [search]
tools = [TavilySearch()]
agent = create_agent(model = llm, tools = tools )

def main():
    result = agent.invoke(
        {"messages":HumanMessage(content="What is the weather in Tokyo?")}
        )
    print(result)
    
# dg

if __name__ == "__main__":
    main()
