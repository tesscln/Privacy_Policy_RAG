from pathlib import Path
from transformers import pipeline
import re

def translate_cnil():
    """
    Translate CNIL text from French to English while preserving structure.
    """
    # Initialize the translation pipeline
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
    
    # Get project root path
    project_root = Path(__file__).parent.parent
    input_path = project_root / "data/CNIL/CNIL.txt"
    output_path = project_root / "data/CNIL/CNIL_english.txt"
    
    try:
        # Read the French text
        print("Reading French text...")
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # Split text into paragraphs while preserving structure
        paragraphs = text.split('\n\n')
        translated_paragraphs = []
        
        print("Translating text while preserving structure...")
        for i, paragraph in enumerate(paragraphs, 1):
            if not paragraph.strip():
                translated_paragraphs.append('')
                continue
                
            # Split paragraph into lines
            lines = paragraph.split('\n')
            translated_lines = []
            
            for line in lines:
                if not line.strip():
                    translated_lines.append('')
                    continue
                    
                # Preserve article numbers and headers
                if re.match(r'^Article \d+', line):
                    translated_lines.append(line)  # Keep original article number
                    continue
                elif line.startswith('NOTA'):
                    translated_lines.append(line)  # Keep NOTA headers
                    continue
                elif re.match(r'^(Titre|Chapitre)', line):
                    # Special handling for titles and chapters
                    parts = re.split(r'(:)', line, maxsplit=1)
                    if len(parts) > 2:
                        header = parts[0] + parts[1]  # Keep original header
                        content = translator(parts[2], max_length=512)[0]['translation_text']
                        translated_lines.append(f"{header} {content}")
                    else:
                        translated_lines.append(line)
                    continue
                
                # Translate the line while preserving any special characters
                try:
                    translated = translator(line, max_length=512)[0]['translation_text']
                    translated_lines.append(translated)
                except Exception as e:
                    print(f"Warning: Could not translate line: {line[:50]}... Error: {str(e)}")
                    translated_lines.append(line)  # Keep original if translation fails
            
            # Rejoin lines with original spacing
            translated_paragraph = '\n'.join(translated_lines)
            translated_paragraphs.append(translated_paragraph)
            
            # Progress update
            if i % 10 == 0:
                print(f"Translated {i}/{len(paragraphs)} paragraphs...")
        
        # Join paragraphs with double newlines to preserve structure
        final_text = '\n\n'.join(translated_paragraphs)
        
        # Write the translated text
        print(f"Writing translated text to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_text)
            
        print("Translation completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    translate_cnil() 