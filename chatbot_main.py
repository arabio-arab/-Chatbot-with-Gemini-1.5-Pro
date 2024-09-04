# Import necessary libraries
import dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import time

# Load environment variables (assuming .env file is in the same directory)
try:
  dotenv.load_dotenv()
except FileNotFoundError:
  print(".env file not found. Please create a file named '.env' with your API key.")
  exit(1)

# Access API key with error handling
try:
  GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
except KeyError:
  print("Missing 'GOOGLE_API_KEY' in .env file. Please add it and try again.")
  exit(1)

# Create LLM instance (assuming other parameters are correct)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=1,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)


instructor_embeddings = HuggingFaceInstructEmbeddings()
vectordb_file_path = "faiss_index"



def create_vector_db():
    # Load data from FAQ sheet
    loader = CSVLoader(file_path='CHATBOT_DATA.csv', source_column="prompt")
    data = loader.load()
    # print(data)


    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=data,embedding=instructor_embeddings)
    
    # Save vector database locally
    vectordb.save_local(vectordb_file_path)
    


def get_qa_chain():
    # Load the vector database from the local folder
    # vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings, allow_dangerous_deserialization=True)


    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True)
    
                                        # chain_type_kwargs={"prompt": PROMPT})

    return chain  



if __name__ == "__main__":
    start_time = time.time()
    create_vector_db()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Le processus a pris {execution_time} secondes.")
    
    chain=get_qa_chain()
    
    print(chain("hello please give me   idea  about any the client for the company "))
