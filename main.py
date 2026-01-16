from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.messages import HumanMessage
# from langgraph.prebuilt import create_react_agent # ReAct agent türünü oluşturuyor
from langgraph.checkpoint.sqlite import SqliteSaver # Kendi agent'ın için çok basit memory tutabilirsin
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor

load_dotenv()

model = ChatOpenAI(model="gpt-4")
memory = SqliteSaver.from_conn_string(":memory:")
search = TavilySearchResults(max_results=2)

prompt = hub.pull("hwchase17/react")

tools=[search]

agent = create_react_agent(model, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, checkpoint=memory) # verbose=True log görmek için

# model_with_tools = model.bind_tools(tools) -> agent_executor ile yaptığımız için buna gerek kalmadı

config = { "configurable": {"thread_id":"abc123"}} # agent önceki konuşmaları hatırlasın diye

if __name__=="__main__":
    # SqliteSaver'ı context manager içinde aç
    '''with SqliteSaver.from_conn_string(":memory:") as memory:
        agent_executor = create_react_agent(
            model,
            tools,
            checkpointer=memory,  # Artık gerçekten BaseCheckpointSaver instance'ı
        )'''

    chat_history=[]

    while True:
        user_input = input(">")
        response = []
        for chunk in agent_executor.stream(
                {"input":user_input,
                    "chat_history":"\n".join(chat_history)
                 },
            config=config
        ):
            if 'text'in chunk:
                print(chunk['text'], end="")
                response.append(chunk['text'])

    chat_history.append(f"AI: {''.join(response)}")

# ReAct makalesini oku
# Adaptive-RAG, Self-RAG, Corrective Retrieval Augmented Generation makalesi oku