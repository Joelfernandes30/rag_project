import os

# =====================================
# CHANGE THESE PATHS
# =====================================

INPUT_FOLDER = r"E:\Gut_Vaidya_Website\rag_model\docs"
OUTPUT_FOLDER = r"E:\Gut_Vaidya_Website\rag_model\chapters"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =====================================
# BOOK STRUCTURE
# =====================================

BOOK_STRUCTURE = {

    "Rasayana": {

        1:(1,34),
        2:(35,52),
        3:(53,85),
        4:(86,95),
        5:(98,115),
        6:(116,125),
        7:(126,142),
        8:(143,157),
        9:(160,171),
        10:(172,177),
        11:(178,187),
        12:(188,202),
        13:(204,222),
        14:(224,248),
        15:(249,270),
        16:(272,285),
        17:(286,300),
        18:(301,318),
        19:(320,333),
        20:(335,340),
        21:(342,356),
        22:(358,363),
        23:(364,389),   # <-- Verify this
        24:(390,407),
        25:(409,418),
        26:(420,454),
        27:(456,460),
        28:(461,533)

    },

    "Pharmaceuticals":{

        1:(537,544),
        2:(545,546),
        3:(547,549),
        4:(550,551),
        5:(552,553),
        6:(554,555),
        7:(556,559),
        8:(564,565),
        9:(566,582)

    },

    "Siddhisthanam":{

        1:(587,871)

    }

}

# =====================================
# CREATE CHAPTERS
# =====================================

for section, chapters in BOOK_STRUCTURE.items():

    section_dir = os.path.join(OUTPUT_FOLDER, section)
    os.makedirs(section_dir, exist_ok=True)

    print(f"\nProcessing {section}")

    for chapter, (start_page, end_page) in chapters.items():

        outfile = os.path.join(
            section_dir,
            f"chapter_{chapter:02d}.md"
        )

        with open(outfile, "w", encoding="utf-8") as out:

            out.write(f"# {section}\n")
            out.write(f"# Chapter {chapter}\n")
            out.write(f"Pages {start_page}-{end_page}\n\n")

            for page in range(start_page, end_page + 1):

                file = os.path.join(
                    INPUT_FOLDER,
                    f"page_{page:04d}.md"
                )

                if not os.path.exists(file):
                    print(f"Missing: page_{page:04d}.md")
                    continue

                with open(file, "r", encoding="utf-8") as f:

                    out.write("\n\n")
                    out.write("=" * 80)
                    out.write(f"\nPAGE {page}\n")
                    out.write("=" * 80)
                    out.write("\n\n")

                    out.write(f.read())

        print(f"✓ Chapter {chapter} created")