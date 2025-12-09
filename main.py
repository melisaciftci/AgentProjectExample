from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent # ReAct agent türünü oluşturuyor
from langgraph.checkpoint.sqlite import SqliteSaver # Kendi agent'ın için çok basit memory tutabilirsin

load_dotenv()

model = ChatOpenAI(model="gpt-4")
search = TavilySearchResults(max_results=2)

tools=[search]

# model_with_tools = model.bind_tools(tools) -> agent_executor ile yaptığımız için buna gerek kalmadı

config = { "configurable": {"thread_id":"abc123"}} # agent önceki konuşmaları hatırlasın diye

if __name__=="__main__":
    # SqliteSaver'ı context manager içinde aç
    with SqliteSaver.from_conn_string(":memory:") as memory:
        agent_executor = create_react_agent(
            model,
            tools,
            checkpointer=memory,  # Artık gerçekten BaseCheckpointSaver instance'ı
        )

        while True:
            user_input = input(">")
            for chunk in agent_executor.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                config=config
            ):
                print(chunk)
                print("--")