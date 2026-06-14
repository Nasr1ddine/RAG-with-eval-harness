from uuid import NAMESPACE_URL, uuid5

from services.ingestion.chunker import Document, HierarchicalChunker


def test_chunk_ids_are_deterministic_from_source_and_position() -> None:
    chunker = HierarchicalChunker(parent_size=50, child_size=20, overlap=5)
    document = Document(
        text="Acme Cloud support is available Monday through Friday from 09:00 to 17:00 UTC.",
        metadata={"source": "acme_policy.md"},
    )

    first_run = chunker.chunk([document])
    second_run = chunker.chunk([document])

    assert first_run[0].parent.id == second_run[0].parent.id
    assert first_run[0].children[0].id == second_run[0].children[0].id
    assert first_run[0].parent.id == str(uuid5(NAMESPACE_URL, "acme_policy.md:parent:0"))
    assert first_run[0].children[0].id == str(
        uuid5(NAMESPACE_URL, "acme_policy.md:parent:0:child:0")
    )
