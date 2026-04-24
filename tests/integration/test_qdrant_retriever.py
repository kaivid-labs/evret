from __future__ import annotations

import pytest

from evret.retrievers import QdrantRetriever

pytestmark = pytest.mark.integration


def test_qdrant_retriever_with_docker(
    require_integration_enabled: None,
    require_docker_daemon: None,
    wait_until_ready_helper,
    unique_collection_name_factory,
) -> None:
    qdrant_client_module = pytest.importorskip("qdrant_client")
    models = pytest.importorskip("qdrant_client.models")
    container_module = pytest.importorskip("testcontainers.core.container")

    DockerContainer = container_module.DockerContainer
    QdrantClient = qdrant_client_module.QdrantClient

    collection_name = unique_collection_name_factory("evret_qdrant")
    with DockerContainer("qdrant/qdrant:v1.9.5").with_exposed_ports(6333) as container:
        qdrant_port = int(container.get_exposed_port(6333))
        qdrant_url = f"http://127.0.0.1:{qdrant_port}"
        client = QdrantClient(url=qdrant_url)

        wait_until_ready_helper(client.get_collections, 45.0)

        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE),
        )
        client.upsert(
            collection_name=collection_name,
            wait=True,
            points=[
                models.PointStruct(
                    id=1,
                    vector=[1.0, 0.0, 0.0],
                    payload={"doc_id": "doc_alpha", "topic": "alpha"},
                ),
                models.PointStruct(
                    id=2,
                    vector=[0.0, 1.0, 0.0],
                    payload={"doc_id": "doc_beta", "topic": "beta"},
                ),
                models.PointStruct(
                    id=3,
                    vector=[0.7, 0.2, 0.0],
                    payload={"doc_id": "doc_mix", "topic": "mixed"},
                ),
            ],
        )

        embedding_lookup = {
            "alpha query": [1.0, 0.0, 0.0],
            "beta query": [0.0, 1.0, 0.0],
        }
        retriever = QdrantRetriever(
            collection_name=collection_name,
            client=client,
            query_encoder=lambda query: embedding_lookup[query],
        )

        results = retriever.retrieve("alpha query", k=2)
        assert len(results) == 2
        assert results[0].doc_id == "doc_alpha"
        assert results[0].metadata["topic"] == "alpha"

        batch_results = retriever.batch_retrieve(["alpha query", "beta query"], k=1)
        assert [rows[0].doc_id for rows in batch_results] == ["doc_alpha", "doc_beta"]
