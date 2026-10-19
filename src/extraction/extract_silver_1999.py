import pymupdf
import re
import pandas as pd
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

PDF_PATH = "data/raw/fidic/fidic-silver-book-2.pdf"

OUTPUT_PATH = "data/processed/fidic_silver_book_1999_clauses.csv"


# Actual General Conditions in this PDF:
# PDF page 18 -> Clause 1
# PDF page 76 -> end of Clause 20
# PDF page 77 -> Appendix
START_PAGE = 18
END_PAGE = 76


# ============================================================
# EXPECTED SILVER BOOK 1999 SUB-CLAUSES
# Taken from the document's Contents.
# ============================================================

EXPECTED_CLAUSES = {
    # Clause 1
    "1.1", "1.2", "1.3", "1.4", "1.5", "1.6",
    "1.7", "1.8", "1.9", "1.10", "1.11", "1.12",
    "1.13", "1.14",

    # Clause 2
    "2.1", "2.2", "2.3", "2.4", "2.5",

    # Clause 3
    "3.1", "3.2", "3.3", "3.4", "3.5",

    # Clause 4
    "4.1", "4.2", "4.3", "4.4", "4.5", "4.6",
    "4.7", "4.8", "4.9", "4.10", "4.11", "4.12",
    "4.13", "4.14", "4.15", "4.16", "4.17", "4.18",
    "4.19", "4.20", "4.21", "4.22", "4.23", "4.24",

    # Clause 5
    "5.1", "5.2", "5.3", "5.4", "5.5", "5.6",
    "5.7", "5.8",

    # Clause 6
    "6.1", "6.2", "6.3", "6.4", "6.5", "6.6",
    "6.7", "6.8", "6.9", "6.10", "6.11",

    # Clause 7
    "7.1", "7.2", "7.3", "7.4", "7.5", "7.6",
    "7.7", "7.8",

    # Clause 8
    "8.1", "8.2", "8.3", "8.4", "8.5", "8.6",
    "8.7", "8.8", "8.9", "8.10", "8.11", "8.12",

    # Clause 9
    "9.1", "9.2", "9.3", "9.4",

    # Clause 10
    "10.1", "10.2", "10.3",

    # Clause 11
    "11.1", "11.2", "11.3", "11.4", "11.5", "11.6",
    "11.7", "11.8", "11.9", "11.10", "11.11",

    # Clause 12
    "12.1", "12.2", "12.3", "12.4",

    # Clause 13
    "13.1", "13.2", "13.3", "13.4", "13.5", "13.6",
    "13.7", "13.8",

    # Clause 14
    "14.1", "14.2", "14.3", "14.4", "14.5", "14.6",
    "14.7", "14.8", "14.9", "14.10", "14.11", "14.12",
    "14.13", "14.14", "14.15",

    # Clause 15
    "15.1", "15.2", "15.3", "15.4", "15.5",

    # Clause 16
    "16.1", "16.2", "16.3", "16.4",

    # Clause 17
    "17.1", "17.2", "17.3", "17.4", "17.5", "17.6",

    # Clause 18
    "18.1", "18.2", "18.3", "18.4",

    # Clause 19
    "19.1", "19.2", "19.3", "19.4",
    "19.5", "19.6", "19.7",

    # Clause 20
    "20.1", "20.2", "20.3", "20.4",
    "20.5", "20.6", "20.7", "20.8",
}


# ============================================================
# HELPERS
# ============================================================

def clean_text(text):
    """
    Normalize whitespace while keeping the clause content readable.
    """

    text = text.replace("\u00ad", "")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_title(title):
    """
    Remove accidental page numbers or noise from title.
    """

    title = clean_text(title)

    title = re.sub(r"\s+\d{1,3}$", "", title)

    return title.strip()


def extract_pages(pdf_path):

    doc = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(doc, start=1):

        text = page.get_text("text")

        pages.append({
            "page": page_number,
            "text": text
        })

    doc.close()

    return pages


# ============================================================
# CLAUSE DETECTION
# ============================================================

def extract_clauses(pages):

    clauses = []

    current = None

    for page in pages:

        page_number = page["page"]

        # Only process actual General Conditions
        if page_number < START_PAGE:
            continue

        if page_number > END_PAGE:
            break

        lines = page["text"].splitlines()

        i = 0

        while i < len(lines):

            line = lines[i].strip()

            if not line:
                i += 1
                continue

            # ------------------------------------------------
            # Detect standalone clause number
            #
            # Example:
            #
            # 8.3
            # Programme
            #
            # 8.4
            # Extension of Time for Completion
            # ------------------------------------------------

            match = re.fullmatch(
                r"(\d{1,2}\.\d{1,2})",
                line
            )

            if match:

                clause_number = match.group(1)

                # IMPORTANT:
                # Only accept known real sub-clauses.
                #
                # This prevents body references such as
                # "20.4" from becoming fake clauses.
                if clause_number not in EXPECTED_CLAUSES:

                    if current:
                        current["clause_text"] += " " + line

                    i += 1
                    continue

                # ------------------------------------------------
                # Save previous clause
                # ------------------------------------------------

                if current:

                    current["clause_text"] = clean_text(
                        current["clause_text"]
                    )

                    if current["clause_text"]:

                        clauses.append(current)

                # ------------------------------------------------
                # Find title
                # ------------------------------------------------

                title_parts = []

                j = i + 1

                while j < len(lines):

                    next_line = lines[j].strip()

                    if not next_line:
                        j += 1
                        continue

                    # Stop if next line is another clause number
                    if re.fullmatch(
                        r"\d{1,2}\.\d{1,2}",
                        next_line
                    ):
                        break

                    # Stop if this looks like a major Clause heading
                    if re.fullmatch(
                        r"\d{1,2}",
                        next_line
                    ):
                        break

                    title_parts.append(next_line)

                    # Titles in this PDF are normally short.
                    # Once we get enough text, treat it as title.
                    if len(" ".join(title_parts)) >= 3:

                        break

                    j += 1

                title = clean_title(
                    " ".join(title_parts)
                )

                current = {
                    "clause_id": (
                        f"SILVER99_{len(clauses) + 1:05d}"
                    ),
                    "contract": "FIDIC Silver Book 1999",
                    "page": page_number,
                    "clause_number": clause_number,
                    "clause_title": title,
                    "clause_text": ""
                }

                # Skip title
                i = max(j + 1, i + 1)

                continue

            # ------------------------------------------------
            # Normal clause text
            # ------------------------------------------------

            if current:

                current["clause_text"] += (
                    " " + line
                )

            i += 1

    # --------------------------------------------------------
    # Save final clause
    # --------------------------------------------------------

    if current:

        current["clause_text"] = clean_text(
            current["clause_text"]
        )

        if current["clause_text"]:

            clauses.append(current)

    return clauses


# ============================================================
# VALIDATION
# ============================================================

def validate_clauses(df):

    extracted = set(
        df["clause_number"].astype(str)
    )

    expected = EXPECTED_CLAUSES

    missing = sorted(
        expected - extracted,
        key=lambda x: (
            int(x.split(".")[0]),
            int(x.split(".")[1])
        )
    )

    unexpected = sorted(
        extracted - expected
    )

    print("\n" + "=" * 60)
    print("SILVER BOOK 1999 VALIDATION")
    print("=" * 60)

    print(
        f"Expected clauses : {len(expected)}"
    )

    print(
        f"Extracted clauses: {len(extracted)}"
    )

    if missing:

        print(
            f"\nMISSING CLAUSES: {len(missing)}"
        )

        print(
            ", ".join(missing)
        )

    else:

        print(
            "\nMissing clauses  : 0"
        )

    if unexpected:

        print(
            f"\nUnexpected clauses: {len(unexpected)}"
        )

        print(
            ", ".join(unexpected)
        )

    else:

        print(
            "Unexpected clauses: 0"
        )

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    Path(
        OUTPUT_PATH
    ).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n" + "=" * 60)
    print("FIDIC SILVER BOOK 1999 EXTRACTION")
    print("=" * 60)

    print(f"PDF        : {PDF_PATH}")
    print(
        f"Page range : {START_PAGE} - {END_PAGE}"
    )

    print("\nReading PDF...")

    pages = extract_pages(
        PDF_PATH
    )

    print(
        f"Pages extracted: {len(pages)}"
    )

    clauses = extract_clauses(
        pages
    )

    df = pd.DataFrame(
        clauses
    )

    if df.empty:

        print(
            "\nERROR: No clauses extracted."
        )

        return

    # Remove accidental duplicate clause numbers
    df = (
        df
        .drop_duplicates(
            subset=["clause_number"],
            keep="first"
        )
        .reset_index(drop=True)
    )

    # Re-create IDs after duplicate removal
    df["clause_id"] = [
        f"SILVER99_{i:05d}"
        for i in range(1, len(df) + 1)
    ]

    # Save
    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\nClauses found: {len(df)}"
    )

    print(
        f"Saved to: {OUTPUT_PATH}"
    )

    print(
        "\nFirst 15 clauses:\n"
    )

    print(
        df[
            [
                "clause_id",
                "page",
                "clause_number",
                "clause_title"
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    # Validation
    validate_clauses(
        df
    )


if __name__ == "__main__":
    main()