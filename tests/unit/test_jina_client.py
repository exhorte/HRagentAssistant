import json

import httpx
import pytest

from hr_assistant.config import Settings
from hr_assistant.embeddings import HashingEmbeddings, JinaEmbeddings, get_embeddings_model


def _client(captured: list, *, status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append({"auth": request.headers.get("authorization"), "body": body})
        data = [{"index": i, "embedding": [float(len(t)), float(i)]} for i, t in enumerate(body["input"])]
        return httpx.Response(status, json={"data": list(reversed(data))})  # server may return rows unordered

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_documents_are_batched_ordered_and_use_the_passage_task():
    calls: list = []
    emb = JinaEmbeddings("test-key", "jina-embeddings-v3", batch_size=2, client=_client(calls))
    vectors = emb.embed_documents(["a", "bb", "ccc"])
    assert vectors == [[1.0, 0.0], [2.0, 1.0], [3.0, 0.0]]
    assert [len(c["body"]["input"]) for c in calls] == [2, 1]
    assert all(c["body"]["task"] == "retrieval.passage" for c in calls)
    assert calls[0]["auth"] == "Bearer test-key"


def test_query_uses_the_query_task_and_older_models_get_no_task():
    calls: list = []
    JinaEmbeddings("k", "jina-embeddings-v3", client=_client(calls)).embed_query("question")
    JinaEmbeddings("k", "jina-embeddings-v2-base-en", client=_client(calls)).embed_query("question")
    assert calls[0]["body"]["task"] == "retrieval.query"
    assert "task" not in calls[1]["body"]


def test_http_errors_propagate_without_the_key_in_the_message():
    emb = JinaEmbeddings("super-secret-key", "jina-embeddings-v3", client=_client([], status=401))
    with pytest.raises(httpx.HTTPStatusError) as exc:
        emb.embed_query("q")
    assert "super-secret-key" not in str(exc.value)


def test_backend_selection():
    assert isinstance(get_embeddings_model(Settings()), HashingEmbeddings)
    selected = get_embeddings_model(Settings(embedding_backend="jina", jina_api_key="k", embedding_model="jina-embeddings-v3"))
    assert isinstance(selected, JinaEmbeddings) and selected.model_name == "jina-embeddings-v3"
