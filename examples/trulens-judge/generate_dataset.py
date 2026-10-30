import json
import os
from pathlib import Path

import pypdfium2 as pdfium
from dotenv import load_dotenv

from evret import DatasetGenerator, SourceDocument

load_dotenv()

HERE = Path(__file__).parent
PDF_PATH = HERE / "data" / "western_digital_2026_annual_report.pdf"
OUTPUT_PATH = HERE / "generated_eval_dataset.json"
MAX_PAGES = 12


def load_pdf_text(pdf_path: Path, max_pages: int | None = None) -> str:
    pdf = pdfium.PdfDocument(str(pdf_path))
    page_count = len(pdf) if max_pages is None else min(max_pages, len(pdf))
    return "\n".join(
        pdf[index].get_textpage().get_text_range() for index in range(page_count)
    )


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY (env or .env) before generating.")

    text = load_pdf_text(PDF_PATH, max_pages=MAX_PAGES)

    generator = DatasetGenerator.from_provider(
        provider="openai",
        model="gpt-6-luna",
        examples_per_chunk=1,
    )
    generated = generator.generate(
        [
            SourceDocument(
                source=PDF_PATH.name,
                text=text,
                metadata={"file_name": PDF_PATH.name},
            )
        ]
    )

    OUTPUT_PATH.write_text(
        json.dumps(generated.to_dict(), indent=2), encoding="utf-8"
    )
    print(
        f"Saved {OUTPUT_PATH.name}: "
        f"{len(generated.chunks)} chunks, {len(generated.examples)} queries"
    )


if __name__ == "__main__":
    main()
