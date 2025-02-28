from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import chromadb

def retrieve_relevant_text(query, top_k=3):
    """Retrieve the most relevant legal text chunks from ChromaDB."""
    chroma_client = chromadb.PersistentClient(path="./chroma_db")  
    embedding_model = OpenAIEmbeddings()
    
    # Load existing Chroma collection
    vector_store = Chroma(client=chroma_client, collection_name="legal_docs", embedding_function=embedding_model)

    # Perform similarity search
    results = vector_store.similarity_search(query, k=top_k)

    return [res.page_content for res in results]

# Example query
query = "What are the user data protection rights under GDPR?"
retrieved_texts = retrieve_relevant_text(query)

print("\n🔹 Top Relevant Texts:")
for idx, text in enumerate(retrieved_texts, 1):
    print(f"\n📄 Chunk {idx}: {text[:500]}...")
