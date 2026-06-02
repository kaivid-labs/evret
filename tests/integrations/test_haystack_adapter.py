import importlib
import sys
import typing
from types import ModuleType
from types import new_class
from typing import Any

import pytest


class Document:
    def __init__(
        self,
        content: str | None = None,
        *,
        id: str | None = None,
        meta: dict[str, Any] | None = None,
        score: float | None = None,
    ) -> None:
        self.content = content
        self.id = id
        self.meta = meta or {}
        self.score = score


class ComponentMeta(type):
    pass


class Component:
    def __call__(self, cls):
        namespace = {
            name: value
            for name, value in cls.__dict__.items()
            if name not in {"__dict__", "__weakref__"}
        }
        namespace["__haystack_component__"] = True
        component_cls = new_class(
            cls.__name__,
            cls.__bases__,
            {"metaclass": ComponentMeta},
            lambda ns: ns.update(namespace),
        )
        original_init = component_cls.__init__

        def __init__(instance, *args, **kwargs):
            typing.get_type_hints(component_cls.run)
            original_init(instance, *args, **kwargs)

        component_cls.__init__ = __init__
        return component_cls

    @staticmethod
    def output_types(**types):
        def decorator(func):
            func.__haystack_output_types__ = types
            return func

        return decorator


class Pipeline:
    def __init__(self) -> None:
        self.components: dict[str, Any] = {}

    def add_component(self, *, instance: Any, name: str) -> None:
        self.components[name] = instance

    def run(self, data: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {name: self.components[name].run(**kwargs) for name, kwargs in data.items()}


haystack_module = ModuleType("haystack")
haystack_module.Document = Document
haystack_module.Pipeline = Pipeline
haystack_module.component = Component()
sys.modules["haystack"] = haystack_module

import evret.integrations.haystack as haystack_integration
from evret.retrievers import BaseRetriever, RetrievalResult
from haystack import Pipeline

haystack_integration = importlib.reload(haystack_integration)
HaystackRetrieverAdapter = haystack_integration.HaystackRetrieverAdapter


class DummyRetriever(BaseRetriever):
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        self.calls.append((query, k))
        return [
            RetrievalResult(
                doc_id="doc_1",
                score=0.87,
                metadata={"document": "alpha content", "source": "unit-test"},
            )
        ]


class DummyHaystackRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def run(self, query: str, top_k: int):
        self.calls.append((query, top_k))
        return {
            "documents": [
                Document(
                    id="doc_1",
                    content="alpha content",
                    meta={"source": "unit-test"},
                    score=0.91,
                ),
                Document(id="doc_2", content="beta content"),
            ]
        }


def test_haystack_adapter_converts_evret_results_to_documents() -> None:
    retriever = DummyRetriever()
    adapter = HaystackRetrieverAdapter(evret_retriever=retriever, k=2)

    result = adapter.run(query="alpha query")

    assert retriever.calls == [("alpha query", 2)]
    docs = result["documents"]
    assert len(docs) == 1
    assert docs[0].id == "doc_1"
    assert docs[0].content == "alpha content"
    assert docs[0].score == 0.87
    assert docs[0].meta["doc_id"] == "doc_1"
    assert docs[0].meta["source"] == "unit-test"


def test_haystack_adapter_runs_inside_pipeline_component() -> None:
    retriever = DummyRetriever()
    adapter = HaystackRetrieverAdapter(evret_retriever=retriever, k=2)
    pipeline = Pipeline()
    pipeline.add_component(instance=adapter, name="retriever")

    result = pipeline.run(data={"retriever": {"query": "alpha query", "top_k": 1}})

    assert retriever.calls == [("alpha query", 1)]
    assert result["retriever"]["documents"][0].content == "alpha content"


def test_haystack_adapter_allows_request_level_top_k_override() -> None:
    retriever = DummyRetriever()
    adapter = HaystackRetrieverAdapter(evret_retriever=retriever, k=4)

    adapter.run(query="beta query", top_k=1)

    assert retriever.calls == [("beta query", 1)]


def test_haystack_adapter_raises_for_non_positive_k() -> None:
    retriever = DummyRetriever()

    with pytest.raises(ValueError, match="k must be a positive integer"):
        HaystackRetrieverAdapter(evret_retriever=retriever, k=0)


def test_haystack_adapter_converts_haystack_documents_to_evret_results() -> None:
    retriever = DummyHaystackRetriever()
    adapter = HaystackRetrieverAdapter(haystack_retriever=retriever)

    results = adapter.retrieve("alpha query", k=1)

    assert retriever.calls == [("alpha query", 1)]
    assert results == [
        RetrievalResult(
            doc_id="doc_1",
            score=0.91,
            metadata={"source": "unit-test", "document": "alpha content"},
        )
    ]
