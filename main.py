from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent # ReAct agent türünü oluşturuyor

load_dotenv()

model = ChatOpenAI(model="gpt-4")

search = TavilySearchResults(max_results=2)

tools=[search]

# model_with_tools = model.bind_tools(tools) -> agent_executor ile yaptığımız için buna gerek kalmadı

agent_executor = create_react_agent(model, tools) # model ve tool birbirine bağlanıyor

if __name__=="__main__":
    response = agent_executor.invoke(
        {"messages": [HumanMessage(content="what is the weather in adana now?")]},
    )
    for r in response["messages"]:
        print(r.content)