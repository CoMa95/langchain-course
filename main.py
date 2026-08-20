from dotenv import load_dotenv
import os

from typing import List
from pydantic import BaseModel, Field

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


# Option B - official tool from the Tavily-LangChain integration
from langchain_tavily import TavilySearch

# format output with pydantic objects
class Source(BaseModel): # object for source, to be part of a nested agent response
    """Schema for a source used by the agent"""
    url:str = Field(description='The URL of the Source')

class AgentResponse(BaseModel):
    """Schema for the agent response with answer and sources"""
    answer:str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description='List of sources used to generate the answer')

# create LLM model, with tools and agents
llm = ChatOpenAI(api_key=os.getenv('OPEN_API_KEY'))
#tools = [search]
tools = [TavilySearch()]
agent = create_agent(model = llm, tools = tools, response_format =  AgentResponse)

def main():
    result = agent.invoke(
        {"messages":HumanMessage(content="What is the weather in Tokyo?")}
        )
    print(result)

if __name__ == "__main__":
    main()
