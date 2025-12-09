import os
import copy
from dotenv import load_dotenv
from utils import get_processed_response_time_slices
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

embedding = AzureOpenAIEmbeddings(
    api_key=os.environ["AZURE_OPENAI_EMBEDDINGS_ADA_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_EMBEDDINGS_ADA_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_EMBEDDINGS_ADA_DEPLOYEMENT_NAME"],
    azure_endpoint=os.environ["AZURE_OPENAI_API_EMBEDDINGS_ADA_ENDPOINT"]
)


def create_vector_store(data: list[dict], max_duration: int | float, vector_db_path: str):
    
    if(os.path.exists(vector_db_path)):
        print("Vector DB already exists!")
        return
    
    processed_data = copy.deepcopy(data)
    
    rag_data = get_processed_response_time_slices(processed_data, max_duration)
    
    rag_documents = [Document(d['text'], metadata = {'start_time': d['start'], 'end_time': d['end']}) for d in rag_data]
    
    store = FAISS.from_documents(
        documents=rag_documents,
        embedding=embedding
    )
    store.save_local(vector_db_path)
    
    print("Vector store created succesfully!")
    print("First document info: ")
    print(f"Size: {len(rag_documents[0].page_content)}")
    print(f"Start: {rag_documents[0].metadata['start_time']}")
    print(f"End: {rag_documents[0].metadata['end_time']}")
    
def load_vector_store(vector_db_path: str):
    
    if not os.path.exists(vector_db_path):
        raise ValueError("Vector db is not present!")
    
    vector_store = FAISS.load_local(
        folder_path=vector_db_path,
        embeddings=embedding,
        allow_dangerous_deserialization=True
    )
    
    return vector_store