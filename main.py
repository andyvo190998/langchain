from typing import List

from pydantic import Field, BaseModel
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langchain_tavily import TavilySearch

tavily = TavilyClient()

load_dotenv()

class Source(BaseModel):
    """Schema for a source used by the agent"""
    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer: str = Field(description="The answer to the query")
    sources: List[Source] = Field(default_factory=list, description="The list of sources used to generate the answer")


@tool
def search(query: str) -> dict:
    """
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Searching for {query} now")
    return tavily.search(query=query)


llm = ChatOpenAI()
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": HumanMessage(content="What is the weather in Munich")})
    print(result['structured_response'])


if __name__ == "__main__":
    main()
