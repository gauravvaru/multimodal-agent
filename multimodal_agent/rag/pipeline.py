import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class EvidenceItem(BaseModel):
    document: str
    page: int | None = None
    chunk: str
    text: str
    retrieval_score: float = Field(ge=0.0)

class RAGPipeline:

    def __init__(self):
        self.documents = []
        self.embeddings = []
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            self.embedder = GoogleGenerativeAIEmbeddings(model='models/text-embedding-004')
        except ImportError:
            self.embedder = None

    def retrieve(self, query: str, *, top_k: int=5) -> list[EvidenceItem]:
        if not self.documents or not self.embedder:
            return []
        try:
            query_embedding = np.array(self.embedder.embed_query(query))
            scores = []
            for doc_emb in self.embeddings:
                norm_query = np.linalg.norm(query_embedding)
                norm_doc = np.linalg.norm(doc_emb)
                if norm_query == 0 or norm_doc == 0:
                    scores.append(0.0)
                else:
                    scores.append(np.dot(query_embedding, doc_emb) / (norm_query * norm_doc))
            top_indices = np.argsort(scores)[-top_k:][::-1]
            results = []
            for idx in top_indices:
                if scores[idx] > 0.3:
                    doc_meta = self.documents[idx]
                    results.append(EvidenceItem(document=doc_meta.get('document', 'session'), page=doc_meta.get('page'), chunk=doc_meta['text'], text=doc_meta['text'], retrieval_score=float(scores[idx])))
            return results
        except Exception as e:
            logger.warning('RAG Retrieval error: %s', e)
            return []

    def index_documents(self, documents: list[dict[str, Any]]) -> None:
        if not self.embedder or not documents:
            return
        self.documents = []
        for doc in documents:
            text = doc.get('text', '')
            if not text:
                continue
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            if not paragraphs:
                paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
            current_chunk = ''
            for p in paragraphs:
                if len(current_chunk) + len(p) < 1000:
                    current_chunk += p + '\n\n'
                else:
                    if current_chunk:
                        chunk_dict = dict(doc)
                        chunk_dict['text'] = current_chunk.strip()
                        self.documents.append(chunk_dict)
                    current_chunk = p + '\n\n'
            if current_chunk:
                chunk_dict = dict(doc)
                chunk_dict['text'] = current_chunk.strip()
                self.documents.append(chunk_dict)
        if not self.documents:
            return
        texts = [doc['text'] for doc in self.documents]
        try:
            emb_results = self.embedder.embed_documents(texts)
            self.embeddings = [np.array(e) for e in emb_results]
        except Exception as e:
            logger.warning('Embedding error: %s', e)
            self.embeddings = []
            self.documents = []