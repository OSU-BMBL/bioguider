
import os
from adalflow import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .common_agent_2step import CommonAgentTwoSteps
from ..rag.rag import RAG

RAG_COLLECT_SYSTEM_PROMPT = ChatPromptTemplate.from_template("""
You are an expert in repository documents retrieval and collection.
Your task is to collect relevant documents based on the user's query using the RAG system.
Here is the user's query:
{query}
The following are the documents extracted from the RAG system:
{documents}
Please analyze the documents one by one and determine which ones are relevant to the user's query.
Return a list of boolean values indicating the relevance of each document. Output example:
[True, False, True, ...]  # True if the document is relevant, False otherwise
""")

class RAGCollectResult(BaseModel):
    """
    Represents the result of a RAG collection task.
    
    Attributes:
        query (str): The user's query.
        documents (list): List of documents retrieved from the RAG system.
        relevance (list): List of boolean values indicating the relevance of each document.
    """
    query: str = Field(..., description="The user's query")
    documents: list[str] = Field(..., description="List of documents retrieved from the RAG system")
    relevance: list[bool] = Field(..., description="List of boolean values indicating the relevance of each document")

RAGCollectResultSchema = {
  'description': "Represents the result of a RAG collection task.\n\nAttributes:\n    query (str): The user's query.\n    documents (list): List of documents retrieved from the RAG system.\n    relevance (list): List of boolean values indicating the relevance of each document.", 
  'properties': {
    'query': {'description': "The user's query", 'title': 'Query', 'type': 'string'}, 
    'documents': {'description': 'List of documents retrieved from the RAG system', 'items': {'type': 'string'}, 'title': 'Documents', 'type': 'array'}, 
    'relevance': {'description': 'List of boolean values indicating the relevance of each document', 'items': {'type': 'boolean'}, 'title': 'Relevance', 'type': 'array'}
  }, 
  'required': [
    'query', 'documents', 'relevance'
  ], 
  'title': 'RAGCollectResult', 
  'type': 'object'
}

class RAGCollectionTaskItem:
    def __init__(self, llm, rag: RAG, step_callback, batch_size: int = 5):
        """
        Initialize the RAGCollectionTaskItem with a repository URL or local path.

        Args:
            rag: An instance of the RAG class
        """
        self.llm = llm
        self.rag = rag
        self.batch_size = batch_size
        self.step_callback = step_callback

    def collect(self, query: str, rag_documents: list[Document]) -> list[Document]:
        relevant_documents = []
        for i in range(0, len(rag_documents), self.batch_size):
            contents = [' - ' + doc.text for doc in rag_documents[i:i + self.batch_size]]
            documents_text = "\n".join(contents)
            prompt = RAG_COLLECT_SYSTEM_PROMPT.format(query=query, documents=documents_text)
            prompt = prompt.replace("{", "{{").replace("}", "}}")  # Escape curly braces for LangChain
            agent = CommonAgentTwoSteps(llm=self.llm)
            res, _, token_usage, reasoning = agent.go(
                system_prompt=prompt,
                instruction_prompt="Please analyze the documents and determine their relevance to the query.",
                schema=RAGCollectResultSchema,
            )
            self.step_callback(
                step_output=f"**Reasoning Process**: {reasoning}\n",
            )
            self.step_callback(
                step_output=f"**RAG Collection Result**: {res}",
            )
            self.step_callback(
                token_usage=token_usage,
            )
            res = RAGCollectResult(**res)
            relevants = self._collect_documents(
                rag_documents[i:i + self.batch_size],
                res.relevance
            )
            relevant_documents.extend(relevants)
        return relevant_documents

    def _collect_documents(self, docs: list[Document], relevants: list[bool]) -> list[Document]:
        """
        Collect documents based on relevance.

        Args:
            docs: List of documents to filter
            relevants: List of boolean values indicating relevance

        Returns:
            List of relevant documents
        """
        return [doc for doc, relevant in zip(docs, relevants) if relevant]
        
            

class RAGCollectionTask:
    def __init__(self, rag: RAG):
        """
        Initialize the RAGCollectionTask with a repository URL or local path.

        Args:
            repo_url_or_path: URL or local path to the repository
            access_token: Optional access token for private repositories
        """
        self.rag = rag
        
    
    def query(self, query: str) -> list:
        """
        Process a query using RAG.

        Args:
            query: The user's query

        Returns:
            retrieved_documents: List of documents retrieved based on the query
        """
        return self.rag.query_doc(query)

