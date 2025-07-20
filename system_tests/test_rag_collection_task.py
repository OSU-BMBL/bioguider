import pytest

from bioguider.rag.rag import RAG
from bioguider.agents.rag_collection_task import RAGCollectionTaskItem, RAGCollectResult

@pytest.mark.skip()
def test_rag_collection_task_item(llm, step_callback):
    json_schema = RAGCollectResult.model_json_schema()
    query = "What is license of BioChatter?"
    rag = RAG()
    rag.initialize_db_manager()
    rag.initialize_repo("https://github.com/biocypher/biochatter.git")
    rag_documents = rag.query_doc(query)[0].documents
    assert rag_documents is not None, "RAG documents should be retrieved successfully."

    task = RAGCollectionTaskItem(llm, rag, step_callback)
    collected_docs = task.collect(query, rag_documents)
    assert collected_docs is not None, "Collected documents should not be None."
    assert len(collected_docs) > 0, "At least one document should be collected."

def test_rag_collection_task_item_1(llm, step_callback):
    json_schema = RAGCollectResult.model_json_schema()
    query = "Hardware and software spec and compatibility description for POPPER"
    rag = RAG()
    rag.initialize_db_manager()
    rag.initialize_repo("https://github.com/snap-stanford/POPPER")
    rag_documents = rag.query_doc(query)[0].documents
    assert rag_documents is not None, "RAG documents should be retrieved successfully."

    task = RAGCollectionTaskItem(llm, rag, step_callback)
    collected_docs = task.collect(query, rag_documents)
    assert collected_docs is not None, "Collected documents should not be None."
    assert len(collected_docs) > 0, "At least one document should be collected."

