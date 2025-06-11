
from typing import List
from adalflow import Document
import pytest
from bioguider.rag.rag import RAG

def extract_source_from_documents(docs: List[Document], doc_scores: List[float]) -> dict:
    extracted_sources = {}
    for ix in range(len(docs)):
        doc = docs[ix]
        if hasattr(doc, 'meta_data') and 'file_path' in doc.meta_data:
            if doc.meta_data['file_path'] not in extracted_sources:
                extracted_sources[doc.meta_data['file_path']] = {
                    'num': 1,
                    'text': [doc.text],
                    'score': [doc_scores[ix]],
                }
            else:
                extracted_sources[doc.meta_data['file_path']]['num'] += 1
                extracted_sources[doc.meta_data['file_path']]['text'].append(doc.text)
                extracted_sources[doc.meta_data['file_path']]['score'].append(doc_scores[ix])
    return extracted_sources

def test_RAG():
    rag = RAG()
    rag.prepare_retriever("https://github.com/snap-stanford/POPPER")
    documents = rag.query_doc("How can I install POPPER?")
    assert documents is not None, "Documents should be retrieved successfully."
    assert len(documents) > 0, "At least one document should be retrieved."

    extracted_sources = extract_source_from_documents(documents[0].documents, documents[0].doc_scores)
    assert len(extracted_sources) > 0, "At least one source should be extracted from the documents."

