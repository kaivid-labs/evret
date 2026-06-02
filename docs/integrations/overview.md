# Integration Overview

Evret provides adapters for common orchestration frameworks.

## LangChain Adapter

`LangChainRetrieverAdapter` wraps an Evret retriever as a LangChain retriever.

```python
from evret.integrations import LangChainRetrieverAdapter

lc_retriever = LangChainRetrieverAdapter(evret_retriever=my_evret_retriever, k=5)
docs = lc_retriever.invoke("what is information retrieval")
```

## LlamaIndex Adapter

`LlamaIndexRetrieverAdapter` wraps an Evret retriever as a LlamaIndex retriever.

```python
from evret.integrations import LlamaIndexRetrieverAdapter

li_retriever = LlamaIndexRetrieverAdapter(evret_retriever=my_evret_retriever, k=5)
```

## Haystack Adapter

`HaystackRetrieverAdapter` wraps an Evret retriever as a Haystack 2.x component.

```python
from haystack import Pipeline
from evret.integrations import HaystackRetrieverAdapter

haystack_retriever = HaystackRetrieverAdapter(evret_retriever=my_evret_retriever, k=5)

pipeline = Pipeline()
pipeline.add_component(instance=haystack_retriever, name="retriever")

result = pipeline.run(data={"retriever": {"query": "what is information retrieval"}})
docs = result["retriever"]["documents"]
```

It can also wrap a Haystack retriever component for Evret evaluation.

```python
from haystack import Document
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from evret.integrations import HaystackRetrieverAdapter

document_store = InMemoryDocumentStore()
document_store.write_documents(
    [
        Document(id="doc_1", content="Information retrieval ranks documents for a query."),
        Document(id="doc_2", content="RAG systems retrieve context before generation."),
    ]
)
haystack_retriever = InMemoryBM25Retriever(document_store=document_store)

evret_retriever = HaystackRetrieverAdapter(haystack_retriever=haystack_retriever)
results = evret_retriever.retrieve("what is information retrieval", k=5)
```

## Dependency Notes

Some integrations require optional packages.

- LangChain adapter needs `langchain`
- LlamaIndex adapter needs `llama-index-core`
- Haystack adapter needs `haystack-ai`

Evret raises `OptionalDependencyError` with install guidance if package is missing.
