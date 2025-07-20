import os
from typing import Any, List, Tuple, Optional, Dict
from uuid import uuid4
import logging
import re
import adalflow as adal
from adalflow.core.types import (
    Conversation,
    DialogTurn,
    UserQuery,
    AssistantResponse,
)
from adalflow.components.retriever.faiss_retriever import FAISSRetriever
from adalflow.components.model_client.openai_client import OpenAIClient
from adalflow.components.model_client.azureai_client import AzureAIClient
from .config import configs, create_model_client, create_model_kwargs
from .data_pipeline import DatabaseManager

logger = logging.getLogger(__name__)

# Maximum token limit for embedding models
MAX_INPUT_TOKENS = 7500  # Safe threshold below 8192 token limit

class RAG(adal.Component):
    """RAG with one repo.
    If you want to load a new repos, call prepare_retriever(repo_url_or_path) first."""

    def __init__(self, use_s3: bool = False):
        """
        Initialize the RAG component.

        Args:
            use_s3: Whether to use S3 for database storage (default: False)
        """
        super().__init__()

        self.embedder = adal.Embedder(
            model_client=create_model_client(),
            model_kwargs=create_model_kwargs(),
        )

        self.initialize_db_manager()

    @property
    def repo_dir(self):
        if self.db_manager:
            return self.db_manager.repo_dir
        return None

    def initialize_db_manager(self):
        """Initialize the database manager with local storage"""
        self.db_manager = DatabaseManager()
        self.transformed_doc_documents: list | None = None
        self.transformed_code_documents: list | None = None
        self.access_token: str | None = None

    def initialize_repo(self, repo_url_or_path: str, access_token: str = None):
        self.repo_url_or_path = repo_url_or_path
        self.access_token = access_token
        self.db_manager.reset_database_and_create_repo(repo_url_or_path, access_token)

    def _prepare_retriever(self):
        """
        Prepare the retriever for a repository.
        Will load database from local storage if available.
        """
        if self.transformed_code_documents is not None and self.transformed_doc_documents is not None:
            # retrievers have been prepared
            return
        self.transformed_doc_documents, self.transformed_code_documents \
            = self.db_manager.prepare_database()
        logger.info(f"Loaded {len(self.transformed_doc_documents)} doc documents for retrieval")
        logger.info(f"Loaded {len(self.transformed_code_documents)} code documents for retrieval")
        self.doc_retriever = FAISSRetriever(
            **configs["retriever"],
            embedder=self.embedder,
            documents=self.transformed_doc_documents,
            document_map_func=lambda doc: doc.vector,
            dimensions=256,
        )
        self.code_retriever = FAISSRetriever(
            **configs["retriever"],
            embedder=self.embedder,
            documents=self.transformed_code_documents,
            document_map_func=lambda doc: doc.vector,
            dimensions=256,
        )

    def query_doc(self, query: str) -> List:
        """
        Process a query using RAG.

        Args:
            query: The user's query

        Returns:
            retrieved_documents: List of documents retrieved based on the query
        """
        self._prepare_retriever()
        retrieved_documents = self.doc_retriever(query)
        # Fill in the documents
        retrieved_documents[0].documents = [
            self.transformed_doc_documents[doc_index]
            for doc_index in retrieved_documents[0].doc_indices
        ]
        return retrieved_documents
    
    def query_code(self, query: str) -> List:
        """
        Process a code query using RAG.

        Args:
            query: The user's code query

        Returns:
            retrieved_documents: List of code documents retrieved based on the query
        """
        try:
            retrieved_documents = self.code_retriever(query)
            # Fill in the documents
            retrieved_documents[0].documents = [
                self.transformed_code_documents[doc_index]
                for doc_index in retrieved_documents[0].doc_indices
            ]
        except Exception as e:
            logger.error(e)
            raise e
        return retrieved_documents
        
    @property
    def save_repo_dir(self) -> str:
        """
        Get the directory where the repository is saved.

        Returns:
            str: The path to the repository directory
        """
        return self.db_manager.repo_paths["save_repo_dir"]