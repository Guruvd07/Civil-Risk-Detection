import pymupdf
import re
import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

PDF_PATH = "data/raw/fidic/FIDIC_Red_Book_1.pdf"
OUTPUT_PATH = "data/processed/red_book_clauses.csv"

START_PAGE = 22
END_PAGE = 127

CONTRACT_NAME = "FIDIC Red Book 2017"


# ============================================================
# CLAUSE NUMBER PATTERN
# ============================================================
#
# Matches ONLY standalone two-level clause numbers:
#
# 1.1
# 1.2
# 1.10
# 8.4
# 20.2
#
# Does NOT match:
#
# 1.1.1
# 1.1.2
# 21.4 [Obtaining DAAB's Decision]
#
# because the number must occupy the complete line.
# ============================================================

SUBCLAUSE_PATTERN = re.compile(
    r"^\s*(\d{1,2}\.\d{1,2})\s*$"
)


# ============================================================
# PAGE EXTRACTION
# ============================================================

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
# TEXT CLEANING
# ============================================================

def clean_text(text):

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_title(title):

    title = clean_text(title)

    # Remove accidental page number at the end
    title = re.sub(r"\s+\d{1,3}$", "", title)

    return title.strip()


# ============================================================
# TITLE DETECTION
# ============================================================

def extract_title(lines, start_index):

    title_parts = []

    j = start_index

    while j < len(lines):

        line = lines[j].strip()

        # Ignore empty lines
        if not line:
            j += 1
            continue

        # Another clause starts
        if SUBCLAUSE_PATTERN.match(line):
            break

        # ----------------------------------------------------
        # First line after clause number = title
        # ----------------------------------------------------

        if not title_parts:

            title_parts.append(line)

            j += 1

            continue

        # ----------------------------------------------------
        # Possible continuation of multi-line title
        # ----------------------------------------------------
        #
        # Example:
        #
        # Notices and
        # Other Communications
        #
        # But we don't want to accidentally consume
        # paragraph text.
        # ----------------------------------------------------

        if (
            len(line.split()) <= 8
            and not line.startswith(
                (
                    "The ",
                    "In ",
                    "Where ",
                    "If ",
                    "Unless ",
                    "For ",
                    "When ",
                    "Any ",
                    "Each ",
                    "This ",
                    "A ",
                    "An ",
                    "As ",
                    "Contractor shall",
                    "Employer shall",
                    "The Contractor",
                    "The Employer",
                )
            )
        ):

            title_parts.append(line)

            j += 1

        else:

            break

    title = clean_title(" ".join(title_parts))

    return title, j


# ============================================================
# CLAUSE EXTRACTION
# ============================================================

def extract_clauses(pages):

    clauses = []

    started = False
    current = None

    for page in pages:

        page_number = page["page"]

        # ----------------------------------------------------
        # MAIN CONTRACT PAGE BOUNDARY
        # ----------------------------------------------------
        #
        # Red Book 2017:
        #
        # Page 22 -> main contract starts
        # Page 127 -> main contract ends
        # Page 128 -> DAA Appendix starts
        #
        # Therefore pages after 127 MUST NOT be processed.
        # ----------------------------------------------------

        if page_number < START_PAGE:
            continue

        if page_number > END_PAGE:
            break

        lines = page["text"].splitlines()

        i = 0

        while i < len(lines):

            line = lines[i].strip()

            # Skip empty lines
            if not line:

                i += 1

                continue

            # ------------------------------------------------
            # DETECT STANDALONE CLAUSE NUMBER
            # ------------------------------------------------

            match = SUBCLAUSE_PATTERN.match(line)

            if match:

                clause_number = match.group(1)

                # ------------------------------------------------
                # START ONLY FROM CLAUSE 1.1
                # ------------------------------------------------

                if not started:

                    if clause_number != "1.1":

                        i += 1

                        continue

                    started = True

                # ------------------------------------------------
                # SAVE PREVIOUS CLAUSE
                # ------------------------------------------------

                if current:

                    current["clause_text"] = clean_text(
                        current["clause_text"]
                    )

                    if len(current["clause_text"]) >= 30:

                        clauses.append(current)

                # ------------------------------------------------
                # EXTRACT TITLE
                # ------------------------------------------------

                title, next_index = extract_title(
                    lines,
                    i + 1
                )

                # ------------------------------------------------
                # CREATE NEW CLAUSE
                # ------------------------------------------------

                current = {

                    "clause_id":
                        f"RED_{len(clauses) + 1:05d}",

                    "contract":
                        CONTRACT_NAME,

                    "page":
                        page_number,

                    "clause_number":
                        clause_number,

                    "clause_title":
                        title,

                    "clause_text":
                        ""
                }

                # Continue after title
                i = next_index

                continue

            # ------------------------------------------------
            # NORMAL CLAUSE TEXT
            # ------------------------------------------------

            if current:

                current["clause_text"] += " " + line

            i += 1

    # ========================================================
    # SAVE FINAL CLAUSE
    # ========================================================

    if current:

        current["clause_text"] = clean_text(
            current["clause_text"]
        )

        if len(current["clause_text"]) >= 30:

            clauses.append(current)

    return clauses


# ============================================================
# VALIDATION
# ============================================================

def validate_clauses(df):

    print("\n" + "=" * 60)
    print("EXTRACTION VALIDATION")
    print("=" * 60)

    print(f"Total clauses extracted : {len(df)}")

    # --------------------------------------------------------
    # Duplicate clause numbers
    # --------------------------------------------------------

    duplicates = df[
        df.duplicated(
            subset=["contract", "clause_number"],
            keep=False
        )
    ]

    print(
        f"Duplicate contract + clause : "
        f"{len(duplicates)} rows"
    )

    if not duplicates.empty:

        print("\nWARNING: Duplicate clauses detected:\n")

        print(
            duplicates[
                [
                    "page",
                    "clause_number",
                    "clause_title"
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # Very short text
    # --------------------------------------------------------

    short_text = df[
        df["clause_text"].str.len() < 30
    ]

    print(
        f"Very short clause text : "
        f"{len(short_text)}"
    )

    # --------------------------------------------------------
    # Page boundary check
    # --------------------------------------------------------

    invalid_pages = df[
        (df["page"] < START_PAGE) |
        (df["page"] > END_PAGE)
    ]

    print(
        f"Clauses outside page boundary : "
        f"{len(invalid_pages)}"
    )

    # --------------------------------------------------------
    # Show clause range
    # --------------------------------------------------------

    if not df.empty:

        print(
            f"First clause : "
            f"{df.iloc[0]['clause_number']}"
        )

        print(
            f"Last clause  : "
            f"{df.iloc[-1]['clause_number']}"
        )

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    Path(OUTPUT_PATH).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("FIDIC RED BOOK 2017 CLAUSE EXTRACTION")
    print("=" * 60)

    print(f"\nPDF          : {PDF_PATH}")
    print(f"Start page   : {START_PAGE}")
    print(f"End page     : {END_PAGE}")
    print(f"Contract     : {CONTRACT_NAME}")

    # --------------------------------------------------------
    # Read PDF
    # --------------------------------------------------------

    print("\nReading PDF...")

    pages = extract_pages(PDF_PATH)

    print(
        f"Pages extracted from PDF : "
        f"{len(pages)}"
    )

    # --------------------------------------------------------
    # Extract clauses
    # --------------------------------------------------------

    clauses = extract_clauses(pages)

    df = pd.DataFrame(clauses)

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if df.empty:

        print("\nERROR: No clauses detected.")

        return

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_clauses(df)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nSaved to:")
    print(OUTPUT_PATH)

    # --------------------------------------------------------
    # Preview
    # --------------------------------------------------------

    print("\nFirst 15 clauses:\n")

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

    # --------------------------------------------------------
    # Last 15 clauses
    # --------------------------------------------------------

    print("\nLast 15 clauses:\n")

    print(
        df[
            [
                "clause_id",
                "page",
                "clause_number",
                "clause_title"
            ]
        ]
        .tail(15)
        .to_string(index=False)
    )

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()