from pathlib import Path
from preprocessing.legal_text_splitter import LegalTextSplitter
from preprocessing.chunk_cnil import chunk_cnil_text

def load_text(file_path):
    """Reads a text file and returns its content."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def load_all_legal_texts():
    """Loads all legal texts and returns them as a dictionary."""
    project_root = Path(__file__).resolve().parent.parent  # Moves two levels up
    
    # Load raw GDPR texts
    gdpr_articles = load_text(project_root / "data/GDPR/gdpr_articles_clean.txt")
    gdpr_recitals = load_text(project_root / "data/GDPR/GDPR_Recitals.txt")
    
    legal_texts = {
        "GDPR_text": gdpr_articles,
        "GDPR_recitals": gdpr_recitals,
    }
    
    # Try to load CNIL text if available
    cnil_path = project_root / "data/CNIL/CNIL_english.txt"
    if cnil_path.exists():
        cnil_text = load_text(cnil_path)
        legal_texts["CNIL_text"] = cnil_text
    else:
        print(f"Warning: CNIL English translation not found at {cnil_path}")
        print("Please run 'python src/translate_cnil.py' first to generate the translation.")
    
    return legal_texts
