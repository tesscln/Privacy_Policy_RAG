from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from pathlib import Path
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress ChromaDB warnings
logging.getLogger('chromadb').setLevel(logging.ERROR)

# Get project root
project_root = Path(__file__).resolve().parent.parent

def get_embedding_model():
    """Initialize the embedding model with error handling."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return HuggingFaceEmbeddings(model_name="nlpaueb/legal-bert-base-uncased")
    except Exception as e:
        logger.error(f"Error initializing embedding model: {str(e)}")
        raise RuntimeError(f"Failed to initialize embedding model: {str(e)}")

def get_chroma_collection():
    """Initialize ChromaDB connection with error handling."""
    try:
        chroma_path = project_root / "chroma_db"
        chroma_path.mkdir(parents=True, exist_ok=True)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chroma_client = chromadb.PersistentClient(path=str(chroma_path))
            return chroma_client.get_collection(name="legal_docs")
    except Exception as e:
        logger.error(f"Error connecting to ChromaDB: {str(e)}")
        raise RuntimeError(f"Failed to connect to ChromaDB: {str(e)}")

# Initialize components
embedding_model = get_embedding_model()
collection = get_chroma_collection()

def retrieve_relevant_text(query, top_k=15):
    """Finds the most relevant legal text chunks given a query."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            query_embedding = embedding_model.embed_query(query)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas"]
            )
            
            if not results['documents'] or not results['documents'][0]:
                logger.warning("No relevant documents found for the query")
                return [{"content": "No relevant legal texts found.", "source": "none"}]
            
            # Combine documents with their sources from metadata
            relevant_texts = []
            source_counts = {"GDPR_text": 0, "GDPR_recitals": 0, "CNIL_text": 0}
            
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                source = metadata.get("source", "unknown")
                relevant_texts.append({
                    "content": doc,
                    "source": source
                })
                if source in source_counts:
                    source_counts[source] += 1
            
            # Log source distribution
            logger.info(f"Retrieved texts by source: {source_counts}")
            
            return relevant_texts
    except Exception as e:
        logger.error(f"Error retrieving relevant text: {str(e)}")
        return [{"content": f"Error retrieving relevant text: {str(e)}", "source": "error"}]

# Example usage
if __name__ == "__main__":
    try:
        query = "What is personal data?"
        relevant_chunks = retrieve_relevant_text(query)
        print("\nRelevant legal texts:")
        for i, chunk in enumerate(relevant_chunks, 1):
            print(f"\n{i}. {chunk['content']} (Source: {chunk['source']})")
    except Exception as e:
        print(f"Error: {str(e)}")
