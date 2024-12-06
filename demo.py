from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
import os
from langchain import hub
from langchain.agents import create_openai_tools_agent
from dotenv import load_dotenv
from langchain.agents import AgentExecutor

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

loader = WebBaseLoader("https://python.langchain.com/v0.1/docs/integrations/tools/edenai_tools/")
docs = loader.load()
print("Documents Loaded:", docs)

documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
print("Documents Split into:", len(documents))

vectordb = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vectordb.as_retriever()
print("Retriever Ready:", retriever)

retriever_tool = create_retriever_tool(retriever, "Edenai_tools", "Search for information about Edenai_tools.")
print("Retriever Tool Configured:", retriever_tool.name)

tools = [retriever_tool]
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

# Assuming prompt is correctly fetched and configured
prompt = hub.pull("hwchase17/openai-functions-agent")
print("Prompt Loaded:", prompt)

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print("Agent Executor:", agent_executor)
response_edenai = agent_executor.invoke({"input": "Tell me about edenai_tools"})
print("Response for 'Tell me about edenai_tools':", response_edenai)

response_use = agent_executor.invoke({"input": "Why we use this tool?"})
print("Response for 'Why we use this tool?':", response_use)