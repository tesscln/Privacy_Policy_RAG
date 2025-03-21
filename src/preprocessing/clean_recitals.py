import re

def clean_gdpr_recitals():
    # Read the file
    try:
        with open('data/GDPR/gdpr_articles_clean.txt', 'r', encoding='utf-8') as file:
            content = file.read()
    except FileNotFoundError:
        print("Error: Could not find the GDPR articles file at data/GDPR/gdpr_articles_clean.txt")
        return

    # Pattern to match "Suitable Recitals" and everything until the next article
    pattern = r'Suitable Recitals\n.*?(?=\d+\. Article \d+:|$)'
    
    # Replace the matched sections with empty string
    cleaned_content = re.sub(pattern, '', content, flags=re.DOTALL)
    
    # Write the cleaned content back to the file
    with open('data/GDPR/gdpr_articles_clean.txt', 'w', encoding='utf-8') as file:
        file.write(cleaned_content)
    
    print("Successfully removed 'Suitable Recitals' sections from the GDPR articles file.")

if __name__ == "__main__":
    clean_gdpr_recitals() 