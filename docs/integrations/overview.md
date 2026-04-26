# Integration Overview

Evret provides adapters for common orchestration frameworks.

## LangChain Adapter

`LangChainRetrieverAdapter` wraps an Evret retriever as a LangChain retriever.

```python
from evret.integrations import LangChainRetrieverAdapter

lc_retriever = LangChainRetrieverAdapter(evret_retriever=my_evret_retriever, k=5)
docs = lc_retriever.invoke("what is rag")
```

## LlamaIndex Adapter

`LlamaIndexRetrieverAdapter` wraps an Evret retriever as a LlamaIndex retriever.

```python
from evret.integrations import LlamaIndexRetrieverAdapter

li_retriever = LlamaIndexRetrieverAdapter(evret_retriever=my_evret_retriever, k=5)
```

## Dependency Notes

Some integrations require optional packages.

- LangChain adapter needs `langchain-core`
- LlamaIndex adapter needs `llama-index-core`

Evret raises `OptionalDependencyError` with install guidance if package is missing.
