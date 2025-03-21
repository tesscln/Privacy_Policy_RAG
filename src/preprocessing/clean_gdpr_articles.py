import re

def clean_gdpr_articles():
    # Read the input file
    with open('data/GDPR/gdpr_articles_clean.txt', 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Remove Ê after article numbers
    cleaned_content = re.sub(r'Article\s+\d+Ê', lambda m: m.group().replace('Ê', ''), content)
    
    # Write the cleaned content back to the file
    with open('data/GDPR/gdpr_articles_clean.txt', 'w', encoding='utf-8') as file:
        file.write(cleaned_content)
    
    print("Cleaned GDPR articles file by removing Ê characters after article numbers.")

if __name__ == "__main__":
    clean_gdpr_articles() 