import pymupdf
import re


PDF_PATH = "data/raw/fidic/FIDICSilver2017-5.pdf"

# Inspect the ending of the main contract
START_PAGE = 115
END_PAGE = 155


def main():

    print("\n" + "=" * 70)
    print("FIDIC SILVER BOOK 2017 - MAIN CONTRACT END DEBUG")
    print("=" * 70)

    doc = pymupdf.open(PDF_PATH)

    pattern = re.compile(
        r"^\s*(\d{1,2}\.\d{1,2})\s*$"
    )

    for page_number in range(
        START_PAGE,
        min(END_PAGE, len(doc)) + 1
    ):

        page = doc[page_number - 1]

        text = page.get_text("text")

        lines = text.splitlines()

        matches = []

        for raw_line in lines:

            line = raw_line.strip()

            if not line:
                continue

            match = pattern.fullmatch(line)

            if match:
                matches.append(match.group(1))

        if matches:

            print(
                f"Page {page_number:3d} -> "
                f"{', '.join(matches)}"
            )

    doc.close()

    print("\n" + "=" * 70)
    print("DEBUG COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()