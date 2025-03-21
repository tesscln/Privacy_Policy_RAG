import pandas as pd
from pathlib import Path

# Load the CSV file
project_root = Path(__file__).resolve().parent.parent  # Moves two levels up

# Define the full path to the CSV file
csv_file = project_root / "data/GDPR/GDPR_recitals.csv"

# Read the CSV
df = pd.read_csv(csv_file)

# Create a text file and write the recitals
output_file = project_root / "data/GDPR/GDPR_Recitals.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for index, row in df.iloc[::-1].iterrows():  # Reverse the order of rows
        recital_number = row.get("Recital_number", "Unknown")
        title = row.get("Recital_title", "No Title")
        content = row.get("Recital_description", "No Content")
        
        f.write(f"{recital_number}: {title}\n")
        f.write(f"{content}\n")
        f.write("=" * 50 + "\n")  # Separator for readability

print(f"Recitals saved to {output_file}")
