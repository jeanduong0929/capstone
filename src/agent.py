import os

from dotenv import load_dotenv

from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, ConversationalChatAgent, load_tools
from langchain.llms import HuggingFaceEndpoint

from tools.docSearchTool import pdf_tool

load_dotenv()

agentSystemTemplate = """
You are a helpful and honest sales assistant for a talent enablement company, Revature. You help customers by suggesting trainings for companies looking to upskill their employees and hire new talent. Your responses are concise, accurate and relevant to the question.
If the user asks for training in a technology not found in sources, connect them with the human representative.
Whenever relevant, tell users why Revature can help them with their talent enablement and upskilling needs. 
"""

llm = HuggingFaceEndpoint(
    endpoint_url=os.environ["LLM_ENDPOINT"],
    task="text2text-generation",
    model_kwargs={"max_new_tokens": 1000},
)

tools = load_tools([], llm=llm)
tools.append(pdf_tool)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

convo_agent = ConversationalChatAgent.from_llm_and_tools(
    llm=llm, tools=tools, system_message=agentSystemTemplate
)
convo_agent_executor = AgentExecutor.from_agent_and_tools(
    agent=convo_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3,
    memory=memory,
)
