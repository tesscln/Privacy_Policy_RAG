import chromadb
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from text_loader import load_and_split_text

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Persistent storage

# Load and split legal text
file_path = "data/gdpr.txt"  # Example: GDPR legal text
chunks = load_and_split_text(file_path)

# Convert text chunks into vector embeddings
embedding_model = OpenAIEmbeddings()
vector_store = Chroma.from_texts(chunks, embedding_model, client=chroma_client, collection_name="legal_docs")

print(f"Stored {len(chunks)} chunks in ChromaDB!")
