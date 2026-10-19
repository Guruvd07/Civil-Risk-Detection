import pymupdf
import re
import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

INPUT_DIR = Path("data/raw/fidic")
OUTPUT_DIR = Path("data/processed")


# ============================================================
# FILE CONFIGURATION
# ============================================================

FILE_CONFIG = {

    # --------------------------------------------------------
    # FIDIC RED BOOK 2017
    # Main Contract:
    # PDF pages 22 -> 127
    # Page 128 starts DAA Appendix
    # --------------------------------------------------------

    "FIDIC_Red_Book_1.pdf": {
        "name": "FIDIC Red Book 2017",
        "category": "primary",
        "start_page": 22,
        "end_page": 127,
        "format": "standard"
    },

    # --------------------------------------------------------
    # FIDIC SILVER BOOK 2017
    # Main Contract:
    # PDF pages 22 -> 123
    # Page 124 starts DAA Appendix
    # --------------------------------------------------------

    "FIDICSilver2017-5.pdf": {
        "name": "FIDIC Silver Book 2017",
        "category": "primary",
        "start_page": 22,
        "end_page": 123,
        "format": "standard"
    },

    # --------------------------------------------------------
    # FIDIC SILVER BOOK 1999
    # Main Contract:
    # PDF pages 18 -> 76
    # --------------------------------------------------------

    "fidic-silver-book-2.pdf": {
        "name": "FIDIC Silver Book 1999",
        "category": "primary",
        "start_page": 18,
        "end_page": 76,
        "format": "silver_1999"
    },

    # --------------------------------------------------------
    # REFERENCE DOCUMENTS
    # --------------------------------------------------------

    "FIDIC_Suite_of_Contracts_4.pdf": {
        "name": "FIDIC Suite of Contracts",
        "category": "reference",
        "format": "reference"
    },

    "FIDIC_guide_final_3.pdf": {
        "name": "FIDIC Guide 2017",
        "category": "reference",
        "format": "reference"
    }
}


# ============================================================
# PATTERNS
# ============================================================

# Matches only two-level contractual clauses:
#
# 1.1
# 1.2
# 8.4
# 20.2
#
# Does NOT match:
#
# 1.1.1
# 1.1.2
# etc.

STANDARD_PATTERN = re.compile(
    r"^\s*(\d{1,2}\.\d{1,2})\s*$"
)


SILVER_1999_PATTERN = re.compile(
    r"^\s*(\d{1,2}\.\d{1,2})\s*$"
)


# ============================================================
# SILVER BOOK 1999 EXPECTED CLAUSES
# ============================================================

EXPECTED_SILVER_1999 = {

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

    text = text.replace("\u00ad", "")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def clean_title(title):

    title = clean_text(title)

    title = re.sub(
        r"\s+\d{1,3}$",
        "",
        title
    )

    return title.strip()


def extract_pages(pdf_path):

    doc = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(doc, start=1):

        pages.append({
            "page": page_number,
            "text": page.get_text("text")
        })

    doc.close()

    return pages


# ============================================================
# STANDARD FIDIC EXTRACTOR
# ============================================================

def extract_standard_clauses(
    pages,
    contract_name,
    start_page,
    end_page
):

    clauses = []

    current = None
    started = False

    for page in pages:

        page_number = page["page"]

        # ----------------------------------------------------
        # PAGE BOUNDARY
        # ----------------------------------------------------

        if page_number < start_page:
            continue

        if page_number > end_page:
            break

        lines = page["text"].splitlines()

        i = 0

        while i < len(lines):

            line = lines[i].strip()

            if not line:

                i += 1
                continue

            match = STANDARD_PATTERN.fullmatch(line)

            if match:

                clause_number = match.group(1)

                # ------------------------------------------------
                # Start only from actual 1.1
                # ------------------------------------------------

                if not started:

                    if clause_number != "1.1":

                        i += 1
                        continue

                    started = True

                # ------------------------------------------------
                # Save previous clause
                # ------------------------------------------------

                if current:

                    current["clause_text"] = clean_text(
                        current["clause_text"]
                    )

                    if len(current["clause_text"]) >= 30:

                        clauses.append(current)

                # ------------------------------------------------
                # Find clause title
                # ------------------------------------------------

                title_parts = []

                j = i + 1

                while j < len(lines):

                    next_line = lines[j].strip()

                    if not next_line:

                        j += 1
                        continue

                    # Another clause begins
                    if STANDARD_PATTERN.fullmatch(next_line):

                        break

                    # First non-empty line = title
                    if not title_parts:

                        title_parts.append(next_line)

                        j += 1
                        continue

                    # Possible multi-line title
                    if (
                        len(next_line.split()) <= 8
                        and not next_line.startswith(
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
                            )
                        )
                    ):

                        title_parts.append(next_line)

                        j += 1

                    else:

                        break

                title = clean_title(
                    " ".join(title_parts)
                )

                # ------------------------------------------------
                # Create clause
                # ------------------------------------------------

                current = {

                    "clause_id": (
                        f"{len(clauses) + 1:05d}"
                    ),

                    "contract": contract_name,

                    "page": page_number,

                    "clause_number": clause_number,

                    "clause_title": title,

                    "clause_text": ""
                }

                # Continue after title
                i = j

                continue

            # ----------------------------------------------------
            # Normal clause text
            # ----------------------------------------------------

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

        if len(current["clause_text"]) >= 30:

            clauses.append(current)

    return clauses


# ============================================================
# SILVER BOOK 1999 EXTRACTOR
# ============================================================

def extract_silver_1999_clauses(
    pages,
    contract_name,
    start_page,
    end_page
):

    clauses = []

    current = None

    for page in pages:

        page_number = page["page"]

        if page_number < start_page:
            continue

        if page_number > end_page:
            break

        lines = page["text"].splitlines()

        i = 0

        while i < len(lines):

            line = lines[i].strip()

            if not line:

                i += 1
                continue

            match = SILVER_1999_PATTERN.fullmatch(line)

            if match:

                clause_number = match.group(1)

                # ------------------------------------------------
                # Only accept expected contractual clauses
                # ------------------------------------------------

                if clause_number not in EXPECTED_SILVER_1999:

                    if current:

                        current["clause_text"] += (
                            " " + line
                        )

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

                    if SILVER_1999_PATTERN.fullmatch(
                        next_line
                    ):

                        break

                    if re.fullmatch(
                        r"\d{1,2}",
                        next_line
                    ):

                        break

                    title_parts.append(next_line)

                    break

                title = clean_title(
                    " ".join(title_parts)
                )

                current = {

                    "clause_id": (
                        f"{len(clauses) + 1:05d}"
                    ),

                    "contract": contract_name,

                    "page": page_number,

                    "clause_number": clause_number,

                    "clause_title": title,

                    "clause_text": ""
                }

                i = max(
                    j + 1,
                    i + 1
                )

                continue

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
# PROCESS ONE FILE
# ============================================================

def process_pdf(
    filename,
    config
):

    pdf_path = INPUT_DIR / filename

    print("\n" + "=" * 60)

    print(
        f"Processing: {filename}"
    )

    print(
        f"Contract  : {config['name']}"
    )

    print(
        f"Category  : {config['category']}"
    )

    if "start_page" in config:

        print(
            f"Page range: "
            f"{config['start_page']} -> "
            f"{config['end_page']}"
        )

    print("=" * 60)

    if not pdf_path.exists():

        print("ERROR: File not found")

        return 0

    # --------------------------------------------------------
    # Reference documents
    # --------------------------------------------------------

    if config["format"] == "reference":

        print(
            "Reference document - clause extraction skipped."
        )

        return 0

    # --------------------------------------------------------
    # Extract PDF pages
    # --------------------------------------------------------

    pages = extract_pages(
        pdf_path
    )

    print(
        f"Pages extracted: {len(pages)}"
    )

    # --------------------------------------------------------
    # Select extractor
    # --------------------------------------------------------

    if config["format"] == "silver_1999":

        clauses = extract_silver_1999_clauses(

            pages,

            config["name"],

            config["start_page"],

            config["end_page"]
        )

    else:

        clauses = extract_standard_clauses(

            pages,

            config["name"],

            config["start_page"],

            config["end_page"]
        )

    # --------------------------------------------------------
    # Empty check
    # --------------------------------------------------------

    if not clauses:

        print(
            "WARNING: No clauses detected"
        )

        return 0

    df = pd.DataFrame(
        clauses
    )

    df["category"] = config["category"]

    df = df[
        [
            "clause_id",
            "contract",
            "category",
            "page",
            "clause_number",
            "clause_title",
            "clause_text"
        ]
    ]

    # --------------------------------------------------------
    # Silver 1999 validation
    # --------------------------------------------------------

    if config["format"] == "silver_1999":

        extracted = set(
            df["clause_number"].astype(str)
        )

        missing = (
            EXPECTED_SILVER_1999
            - extracted
        )

        unexpected = (
            extracted
            - EXPECTED_SILVER_1999
        )

        print(
            f"Expected clauses : "
            f"{len(EXPECTED_SILVER_1999)}"
        )

        print(
            f"Extracted clauses: "
            f"{len(extracted)}"
        )

        print(
            f"Missing clauses  : "
            f"{len(missing)}"
        )

        print(
            f"Unexpected clauses: "
            f"{len(unexpected)}"
        )

        if missing:

            print(
                "Missing:",
                sorted(missing)
            )

        if unexpected:

            print(
                "Unexpected:",
                sorted(unexpected)
            )

    # --------------------------------------------------------
    # Duplicate validation
    # --------------------------------------------------------

    duplicate_mask = df.duplicated(
        subset=[
            "contract",
            "clause_number"
        ],
        keep=False
    )

    duplicate_rows = int(
        duplicate_mask.sum()
    )

    duplicate_groups = int(
        df.loc[
            duplicate_mask,
            [
                "contract",
                "clause_number"
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    short_text = int(
        (
            df["clause_text"]
            .str.len()
            < 30
        ).sum()
    )

    outside_boundary = int(
        (
            (df["page"] < config["start_page"])
            |
            (df["page"] > config["end_page"])
        ).sum()
    )

    print("\n" + "=" * 60)

    print(
        "EXTRACTION VALIDATION"
    )

    print("=" * 60)

    print(
        f"Total clauses       : {len(df)}"
    )

    print(
        f"Duplicate rows      : {duplicate_rows}"
    )

    print(
        f"Duplicate groups    : {duplicate_groups}"
    )

    print(
        f"Very short text     : {short_text}"
    )

    print(
        f"Outside page range  : {outside_boundary}"
    )

    if len(df) > 0:

        print(
            f"First clause        : "
            f"{df.iloc[0]['clause_number']}"
        )

        print(
            f"Last clause         : "
            f"{df.iloc[-1]['clause_number']}"
        )

    print("=" * 60)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_name = Path(
        filename
    ).stem.lower()

    output_path = (
        OUTPUT_DIR
        /
        f"{output_name}_clauses.csv"
    )

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\nClauses found : {len(df)}"
    )

    print(
        f"Saved to      : {output_path}"
    )

    # --------------------------------------------------------
    # First 10
    # --------------------------------------------------------

    print(
        "\nFirst 10 clauses:"
    )

    print(
        df[
            [
                "clause_number",
                "clause_title"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Last 10
    # --------------------------------------------------------

    print(
        "\nLast 10 clauses:"
    )

    print(
        df[
            [
                "clause_number",
                "clause_title"
            ]
        ]
        .tail(10)
        .to_string(index=False)
    )

    return len(df)


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "\nFIDIC BATCH CLAUSE EXTRACTION"
    )

    print("=" * 60)

    results = []

    total_clauses = 0

    for filename, config in FILE_CONFIG.items():

        count = process_pdf(
            filename,
            config
        )

        total_clauses += count

        results.append({

            "file": filename,

            "contract": config["name"],

            "category": config["category"],

            "clauses": count
        })

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        results
    )

    summary_path = (
        OUTPUT_DIR
        /
        "fidic_extraction_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 60)

    print(
        "BATCH EXTRACTION COMPLETE"
    )

    print("=" * 60)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        f"\nTotal clauses extracted: "
        f"{total_clauses}"
    )

    print(
        f"Summary saved to: "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()