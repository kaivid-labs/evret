import json
from pathlib import Path
from evret import DatasetGenerator, SourceDocument

import os
os.environ['OPENAI_API_KEY'] = "<replace-with-your-api-key>"

PDF_PATH = Path(__file__).with_name("react_agent_paper.pdf")
OUTPUT_PATH = Path("generated_eval_dataset.json")

def load_pdf_text(pdf_path: Path) -> str:
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(pdf_path)
    return "\n".join(page.get_textpage().get_text_range() for page in pdf)

def main() -> None:
    text = load_pdf_text(PDF_PATH)
    generator = DatasetGenerator.from_provider(
        provider="openai",
        model="gpt-5.4-nano",
        examples_per_chunk=2,
    )

    generated = generator.generate(
        [
            SourceDocument(
                source=str(PDF_PATH),
                text=text,
                metadata={"file_name": PDF_PATH.name},
            )
        ]
    )
    OUTPUT_PATH.write_text(json.dumps(generated.to_dict(), indent=2),encoding="utf-8")
    print(f"Saved {OUTPUT_PATH}")

if __name__ == "__main__":
    main()