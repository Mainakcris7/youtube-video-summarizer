import os
import copy
from dotenv import load_dotenv
from utils import get_grouped_data
from langchain_openai import AzureOpenAIEmbeddings
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

# embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
embedding = AzureOpenAIEmbeddings(
    api_key=os.environ["AZURE_OPENAI_EMBEDDINGS_ADA_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_EMBEDDINGS_ADA_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_EMBEDDINGS_ADA_DEPLOYEMENT_NAME"],
    azure_endpoint=os.environ["AZURE_OPENAI_API_EMBEDDINGS_ADA_ENDPOINT"]
)


def create_vector_store(data: list[dict], max_duration: int | float, vector_db_path: str):
    """
    Creates a persistent vector database from time-based, segmented transcription data.

    The function intelligently **merges adjacent segments** of `data` (which have 'start', 
    'end', and 'text') to form larger, contextually coherent chunks, ensuring 
    that the *resulting chunk duration* does not exceed `max_duration` (in seconds).
    It then embeds the text of these new chunks and saves the vector index to 
    `vector_db_path` for efficient semantic search.

    Args:
        data: List of dictionaries; each must contain 'start', 'end', and 'text' 
              keys representing a small segment.
        max_duration: The maximum allowed duration (in seconds) for a merged 
                      segment (chunk) to retain context while ensuring search 
                      relevance.
        vector_db_path: File path to save the resulting vector database.
    """
    if (os.path.exists(vector_db_path)):
        print("Vector DB already exists!")
        return

    processed_data = copy.deepcopy(data)

    print("Creating vector store...")
    rag_data = get_grouped_data(processed_data, max_duration)

    rag_documents = [Document(d['text'], metadata={
                              'start_time': d['start'], 'end_time': d['end']}) for d in rag_data]

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


def load_vector_store(vector_db_path: str) -> FAISS:
    """
    Loads a persisted FAISS vector index from the local path.

    Checks if the `vector_db_path` exists, then loads the FAISS index 
    using the provided embedding function, enabling deserialization.

    Args:
        vector_db_path: Folder path where the FAISS store files are saved.

    Returns:
        The loaded FAISS vector store instance.

    Raises:
        ValueError: If the directory at `vector_db_path` is not found.
    """
    if not os.path.exists(vector_db_path):
        raise ValueError("Vector db is not present!")

    vector_store = FAISS.load_local(
        folder_path=vector_db_path,
        embeddings=embedding,
        allow_dangerous_deserialization=True
    )

    return vector_store
