from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_text(file_path, chunk_size=1000, chunk_overlap=200):
    """Loads a text file and splits it into overlapping chunks."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Create a text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )

    # Split text into chunks
    chunks = text_splitter.split_text(text)
    
    return chunks
