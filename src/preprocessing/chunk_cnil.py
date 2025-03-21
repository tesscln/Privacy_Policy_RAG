from pathlib import Path
from preprocessing.legal_text_splitter import LegalTextSplitter

def chunk_cnil_text():
    """
    Chunk the CNIL text by articles and save the chunks.
    """
    # Get the project root path (two levels up from this file)
    project_root = Path(__file__).resolve().parent.parent.parent
    
    # Initialize the text splitter
    splitter = LegalTextSplitter()
    
    # Read the CNIL text file
    input_path = project_root / "data/CNIL/CNIL_english.txt"
    output_path = project_root / "data/CNIL/CNIL_chunks.txt"
    
    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            print(f"Trying to read file with {encoding} encoding...")
            with open(input_path, 'r', encoding=encoding) as f:
                text = f.read()
                print(f"Successfully read file with {encoding} encoding")
                print(f"First 500 characters of text:\n{text[:500]}")
                break
        except UnicodeDecodeError:
            print(f"Failed to read with {encoding} encoding")
            continue
    else:
        print("Failed to read file with any encoding")
        return
    
    try:
        # Split the text into chunks
        print("Splitting text into chunks...")
        chunks = splitter.split_text(text)
        
        # Write chunks to output file
        print(f"Writing {len(chunks)} chunks to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks, 1):
                f.write(f"Chunk {i}:\n{chunk}\n\n{'='*80}\n\n")
        
        print(f"Successfully created {len(chunks)} chunks from CNIL text")
        return chunks  # Return the chunks for use in store_embeddings.py
        
    except Exception as e:
        print(f"Error processing CNIL text: {str(e)}")
        return []

if __name__ == "__main__":
    chunk_cnil_text() 