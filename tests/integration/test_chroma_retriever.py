from __future__ import annotations

import pytest

from evret.retrievers import ChromaRetriever

pytestmark = pytest.mark.integration


def test_chroma_retriever_with_docker(
    require_integration_enabled: None,
    require_docker_daemon: None,
    wait_until_ready_helper,
    unique_collection_name_factory,
) -> None:
    chromadb = pytest.importorskip("chromadb")
    container_module = pytest.importorskip("testcontainers.core.container")
    DockerContainer = container_module.DockerContainer

    collection_name = unique_collection_name_factory("evret_chroma")
    with DockerContainer("chromadb/chroma:0.5.5").with_exposed_ports(8000) as container:
        chroma_port = int(container.get_exposed_port(8000))
        client = chromadb.HttpClient(host="127.0.0.1", port=chroma_port)

        wait_until_ready_helper(client.heartbeat, 60.0)

        collection = client.get_or_create_collection(name=collection_name)
        collection.add(
            ids=["doc_alpha", "doc_beta", "doc_mix"],
            documents=["alpha text", "beta text", "mixed text"],
            metadatas=[
                {"topic": "alpha"},
                {"topic": "beta"},
                {"topic": "mixed"},
            ],
            embeddings=[
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.6, 0.3, 0.0],
            ],
        )

        embedding_lookup = {
            "alpha query": [1.0, 0.0, 0.0],
            "beta query": [0.0, 1.0, 0.0],
        }
        retriever = ChromaRetriever(
            collection_name=collection_name,
            client=client,
            query_encoder=lambda query: embedding_lookup[query],
        )

        results = retriever.retrieve("alpha query", k=2)
        assert len(results) == 2
        assert results[0].doc_id == "doc_alpha"
        assert results[0].metadata["topic"] == "alpha"
        assert results[0].metadata["document"] == "alpha text"

        batch_results = retriever.batch_retrieve(["alpha query", "beta query"], k=1)
        assert [rows[0].doc_id for rows in batch_results] == ["doc_alpha", "doc_beta"]
