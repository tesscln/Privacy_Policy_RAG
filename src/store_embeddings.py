from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from text_loader import load_all_legal_texts
from pathlib import Path
from preprocessing.legal_text_splitter import LegalTextSplitter
from preprocessing.chunk_cnil import chunk_cnil_text

# Get project root
project_root = Path(__file__).resolve().parent.parent

# Load legal texts
legal_texts = load_all_legal_texts()

# Initialize text splitter for GDPR
text_splitter = LegalTextSplitter()

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="nlpaueb/legal-bert-base-uncased")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path=str(project_root / "chroma_db"))

# Delete existing collection if it exists
try:
    chroma_client.delete_collection(name="legal_docs")
except:
    pass

# Create new collection
collection = chroma_client.create_collection(name="legal_docs")

# Process each legal document
for source_name, text in legal_texts.items():
    print(f"\nProcessing {source_name}...")
    print(f"Text length: {len(text)} characters")
    
    # Split text into chunks using the legal text splitter
    chunks = text_splitter.split_text(text)
    print(f"Created {len(chunks)} chunks for {source_name}")
    print(f"First chunk preview: {chunks[0][:200]}...")
    
    chunk_count = 0
    # Store each chunk with its source metadata
    for i, chunk in enumerate(chunks):
        if not chunk.strip():  # Skip empty chunks
            continue
            
        try:
            embedding = embedding_model.embed_query(chunk)  # Convert text into vector
            collection.add(
                ids=[f"{source_name}-{i}"],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{"source": source_name}]  # Add source metadata
            )
            chunk_count += 1
        except Exception as e:
            print(f"Error processing chunk {i} from {source_name}: {str(e)}")
            continue
    
    print(f"Successfully stored {chunk_count} chunks for {source_name}")

print("\nLegal texts embedded and stored in ChromaDB!")
# Print collection stats
print("\nCollection statistics:")
collection_stats = collection.count()
print(f"Total documents in collection: {collection_stats}")
