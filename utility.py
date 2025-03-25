from bs4 import BeautifulSoup
import json

# Read the file content
with open("roots.txt", "r", encoding="utf-8") as file:
    html_content = file.read()

# Parse with BeautifulSoup
soup = BeautifulSoup(html_content, "html.parser")

# Extract all table rows
rows = soup.find_all("tr")

# Prepare the output structure
output = []
unique_id = 1

for row in rows:
    cols = row.find_all("td")
    if len(cols) < 5:
        continue  # skip malformed rows

    root = cols[1].get_text(strip=True)
    meaning = cols[2].get_text(strip=True)
    examples_raw = cols[4].get_text(separator=" ", strip=True)

    # Split examples by ';', then further split into word and meaning
    examples = []
    for ex in examples_raw.split(";"):
        if "-" in ex:
            word, word_meaning = ex.split("-", 1)
            examples.append((word.strip(), word_meaning.strip()))

    output.append({
        "id": unique_id,
        "root": root,
        "meaning": meaning,
        "examples": examples
    })

    unique_id += 1

# Save to JSON file
with open("roots.json", "w", encoding="utf-8") as json_file:
    json.dump(output, json_file, indent=2, ensure_ascii=False)

output[:3]  # Show a preview of the first 3 entries
