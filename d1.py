import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from dotenv import load_dotenv
import os
from langchain import hub

# Load environment variables and set API key
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# Initialize the LLM with a specific model
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

# Function to set up the retriever with documents from a specified webpage
def setup_retriever(webpage_link, webpage_name):
    loader = WebBaseLoader(webpage_link)
    docs = loader.load()
    documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    vectordb = FAISS.from_documents(documents, OpenAIEmbeddings())
    retriever = vectordb.as_retriever()
    tool_name = f"{webpage_name} - WebInfoTool"
    description = f"Search for information on {webpage_name}."
    retriever_tool = create_retriever_tool(retriever, tool_name, description)
    return retriever_tool

# Home page setup in Streamlit
def home():
    st.title("Home Page")
    with st.form(key='input_form'):
        webpage_name = st.text_input("Enter webpage name:")
        webpage_link = st.text_input("Enter website link:")
        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        retriever_tool = setup_retriever(webpage_link, webpage_name)
        prompt = hub.pull("hwchase17/openai-functions-agent")  # Fetching the latest prompt configuration
        agent = create_openai_tools_agent(llm, [retriever_tool], prompt)
        st.session_state['agent_executor'] = AgentExecutor(agent=agent, tools=[retriever_tool], verbose=True)
        st.session_state['webpage_name'] = webpage_name
        st.session_state['webpage_link'] = webpage_link
        st.success("Setup complete! Go to the Display page to interact.")

# Display page setup in Streamlit
def display():
    st.title("Display Page")
    if 'agent_executor' in st.session_state:
        user_query = st.text_input("Ask a question about the website:")
        if st.button("Get Answer"):
            response = st.session_state['agent_executor'].invoke({"input": user_query})
            st.write("Answer: ", response['output'])
    else:
        st.error("No data to display. Please set up the website on the Home page first.")

# Sidebar navigation for Streamlit pages
page = st.sidebar.selectbox('Choose a page', ['Home', 'Display'])

if page == 'Home':
    home()
elif page == 'Display':
    display()
