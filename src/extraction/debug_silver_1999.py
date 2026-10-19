import pymupdf
import re

PDF_PATH = "data/raw/fidic/fidic-silver-book-2.pdf"

PATTERN = re.compile(
    r"Sub-Clause\s+(\d{1,2}\.\d{1,2})",
    re.IGNORECASE
)


def main():

    doc = pymupdf.open(PDF_PATH)

    clauses = []

    for page_number, page in enumerate(doc, start=1):

        text = page.get_text("text")

        matches = PATTERN.findall(text)

        for clause in matches:

            clauses.append({
                "page": page_number,
                "clause": clause
            })

    doc.close()

    # Remove duplicates while keeping order
    unique = []
    seen = set()

    for item in clauses:

        key = item["clause"]

        if key not in seen:

            seen.add(key)
            unique.append(item)

    print("\n" + "=" * 60)
    print("SILVER BOOK 1999 - DETECTED CLAUSES")
    print("=" * 60)

    for item in unique:

        print(
            f"Page {item['page']:>3}  ->  "
            f"Sub-Clause {item['clause']}"
        )

    print("\n" + "=" * 60)
    print(f"Unique clauses detected: {len(unique)}")
    print("=" * 60)


if __name__ == "__main__":
    main()