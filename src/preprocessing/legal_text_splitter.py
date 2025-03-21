import re
from typing import List
from langchain.text_splitter import TextSplitter

class LegalTextSplitter(TextSplitter):
    """Custom text splitter for legal documents that splits by articles and recitals."""
    
    def __init__(self):
        # Patterns to match article and recital headers
        self.article_patterns = [
            re.compile(r'^\s*\d+\.\s*Article\s+\d+[\.:]\s*', re.MULTILINE),  # For "1. Article 1:"
            re.compile(r'^\s*Article\s+\d+[\.:]\s*', re.MULTILINE),          # For "Article 1:"
            re.compile(r'^\s*\d+\.\s*Article\s+\d+\s*', re.MULTILINE),       # For "1. Article 1"
            re.compile(r'^\s*Article\s+\d+\s*', re.MULTILINE),               # For "Article 1"
            re.compile(r'^\s*Art\.\s*\d+[\.:]\s*', re.MULTILINE),            # For "Art. 1:"
            re.compile(r'^\s*Art\.\s*\d+\s*', re.MULTILINE)                  # For "Art. 1"
        ]
        
        self.recital_patterns = [
            re.compile(r'^\s*Recital\s+\d+[\.:]\s*', re.MULTILINE),
            re.compile(r'^\s*Recital\s+\d+\.\s*', re.MULTILINE),
            re.compile(r'^\s*Recital\s+\d+\s*', re.MULTILINE),
            re.compile(r'^\s*\(\d+\)\s+', re.MULTILINE)  # For "(1) ..."
        ]
        
        # Pattern to match section headers
        self.section_pattern = re.compile(r'^\s*[IVX]+\.\s+[A-Za-z\s]+$', re.MULTILINE)
        
    def is_header(self, line: str) -> bool:
        """Check if a line is a header (article, recital, or section)."""
        line = line.strip()
        
        # Check for article headers
        if any(pattern.match(line) for pattern in self.article_patterns):
            return True
            
        # Check for recital headers
        if any(pattern.match(line) for pattern in self.recital_patterns):
            return True
            
        # Check for section headers
        if self.section_pattern.match(line):
            return True
            
        return False
        
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks based on articles, recitals, and sections."""
        chunks = []
        current_chunk = []
        
        # Split text into lines
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            # Skip empty lines at the start of chunks
            if not line and not current_chunk:
                continue
                
            # Check if line is a header
            if self.is_header(line):
                # If we have content in current chunk, save it
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
            
            # Add line to current chunk
            if line:
                current_chunk.append(line)
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        # Clean up chunks
        cleaned_chunks = []
        for chunk in chunks:
            # Remove empty lines at start/end and normalize internal whitespace
            cleaned_lines = [line.strip() for line in chunk.split('\n')]
            cleaned_lines = [line for line in cleaned_lines if line]
            if cleaned_lines:
                cleaned_chunks.append('\n'.join(cleaned_lines))
        
        return cleaned_chunks
    
    def split_documents(self, documents):
        """Split documents into chunks based on articles and recitals."""
        # This method is required by the TextSplitter interface
        # but we don't need it for our use case
        raise NotImplementedError("split_documents is not implemented for LegalTextSplitter") 