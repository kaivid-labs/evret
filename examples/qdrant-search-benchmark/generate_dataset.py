from __future__ import annotations

import json
import os

from common import DATASET_PATH, answerable_category_counts, load_documents
from evret import DatasetGenerator, SourceDocument

from dotenv import load_dotenv
load_dotenv()

EXAMPLES_PER_CHUNK = 3
MODEL = "gpt-6-luna"


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY (env or .env) before generating.")

    documents = load_documents()
    print(f"Loaded {len(documents)} documents:")
    for name, text in documents:
        print(f"  {name}: {len(text.split())} words")

    generator = DatasetGenerator.from_provider(
        provider="openai",
        model=MODEL,
        examples_per_chunk=EXAMPLES_PER_CHUNK,
    )
    generated = generator.generate(
        [
            SourceDocument(source=name, text=text, metadata={"file_name": name})
            for name, text in documents
        ]
    )

    raw = generated.to_dict()
    DATASET_PATH.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    answerable = answerable_category_counts(raw)
    print(f"\nSaved {DATASET_PATH.name}")
    print(f"  chunks (documents): {len(raw['documents'])}")
    print(f"  queries total:      {len(raw['queries'])}")
    print(f"  answerable queries: {sum(answerable.values())}")
    for category, count in sorted(answerable.items()):
        print(f"    {category:16s} {count}")


if __name__ == "__main__":
    main()
