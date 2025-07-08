from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langgraph.graph import START,END,StateGraph
from langgraph.prebuilt import ToolNode,tools_condition
import os

load_dotenv()

loader = PyMuPDFLoader("apjspeech.pdf")
docs = loader.load()

#text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500,chunk_overlap=50)
chunks=text_splitter.split_documents(docs)

#embeddings 
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

#vector Store
db = FAISS.from_documents(chunks,embeddings)

#retrieve tools
retriever_tool = create_retriever_tool(
    retriever = db.as_retriever(),
    name = "document Search",
    description = "Full information about APJ abdul kalam speech.")


#state class
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

#load model
llm = ChatGroq(model = "llama3-8b-8192")

#bind tools
llm_with_tools = llm.bind_tools([retriever_tool])

#message invoke for state
def chatbot(state:State) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages":[response]}

#graphbuilder

graph_builder = StateGraph(State)

#add node
graph_builder.add_node("chatbot",chatbot)

#add tool node
tool_node = ToolNode(tools=[retriever_tool])
graph_builder.add_node("tools",tool_node)

#add conditional edge
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition
)

#add edge
graph_builder.add_edge("tools", "chatbot")  
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")
graph = graph_builder.compile()

#chatbot input-output
chat_history = []
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    chat_history.append({"role": "user", "content": user_input})
    result = graph.invoke({"messages": chat_history})
    reply = result["messages"][-1]
    chat_history.append(reply)
    print("Bot:", reply.content)


