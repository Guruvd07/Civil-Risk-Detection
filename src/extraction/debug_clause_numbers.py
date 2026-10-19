import pymupdf
import re


PDF_PATH = "data/raw/fidic/FIDIC_Red_Book_1.pdf"

# We are specifically inspecting Clause 1 area
START_PAGE = 22
END_PAGE = 40


def main():

    print("\n" + "=" * 60)
    print("FIDIC RED BOOK - CLAUSE NUMBER DEBUG")
    print("=" * 60)

    doc = pymupdf.open(PDF_PATH)

    pattern = re.compile(
        r"^\s*(\d{1,2}\.\d{1,2})\s*$"
    )

    for page_number in range(
        START_PAGE,
        END_PAGE + 1
    ):

        page = doc[page_number - 1]

        text = page.get_text("text")

        lines = text.splitlines()

        print(
            f"\n--- PAGE {page_number} ---"
        )

        for index, raw_line in enumerate(lines):

            line = raw_line.strip()

            if not line:
                continue

            match = pattern.fullmatch(line)

            if match:

                print(
                    f"MATCH: '{line}' "
                    f"-> {match.group(1)}"
                )

            # Print suspicious lines containing
            # 1.1 / 1.10 / 1.11 etc.
            elif re.search(
                r"1\.\d+",
                line
            ):

                print(
                    f"SUSPECT: '{line}'"
                )

    doc.close()

    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()