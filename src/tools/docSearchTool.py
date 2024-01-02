from dotenv import load_dotenv

from langchain.embeddings import OpenAIEmbeddings
from langchain.tools import Tool
from langchain.vectorstores import Pinecone
import os
import pinecone
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()

embeddings_model = OpenAIEmbeddings(deployment="demo-text-embed")

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
pinecone.init(
	api_key=PINECONE_API_KEY,     
	environment= PINECONE_ENV     
)

index_name= "demo-index"

index = pinecone.Index(index_name)
docsearch = Pinecone(index, embeddings_model.embed_query, "text")

docSearchTemplate="""
You are an intelligent, supportive and honest assistant. Don't provide a wrong answer if you are unsure of the context. Answer the question to the best of your ability.
Begin!
Question: {question}
Answer:"""

llm = AzureChatOpenAI(deployment_name="demo-models", model_name="gpt-35-turbo")
prompt = PromptTemplate(template=docSearchTemplate, input_variables=["question"])
docsearchChain = LLMChain(prompt=prompt, llm = llm)

pdf_tool = Tool(
    name='pdf retrieval tool',
    func=docsearchChain.run,
    retriever=docsearch,
    description="Used to retrieve Java Fullstack and SRE curricula information from the pdfs",
    retriever_top_k=3
)