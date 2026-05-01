# from flask import Flask, render_template, request
# from src.helper import get_embedding
# from langchain_openai import ChatOpenAI
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
# from langchain_core.prompts import ChatPromptTemplate
# from dotenv import load_dotenv
# from src.prompt import *
# import os
# from store import docs


from langchain_groq import ChatGroq 
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings # Isko change kiya
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from flask import Flask, render_template, request
from src.helper import get_embedding
from src.prompt import *
from store import docs


app = Flask(__name__)



load_dotenv()


GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
os.environ["GROQ_API_KEY"] =GROQ_API_KEY

embeddings = get_embedding()


retriever = docs.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatGroq(
    temperature=0.6,
    model_name="llama-3.1-8b-instant",  # ya "mixtral-8x7b-32768"
    
)


# Create prompt templates
base_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])


# Create chains
question_answer_chain = create_stuff_documents_chain(llm, base_prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    response = rag_chain.invoke({"input": msg})
    
    print("Response: ", response["answer"])
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)