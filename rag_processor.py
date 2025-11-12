"""
RAG (Retrieval-Augmented Generation) System for Shopfloor Data
Luke's implementation for WFC AI Integration

This module provides:
1. Document processing and chunking from CSV data
2. Vector embeddings using sentence transformers
3. Semantic search and retrieval
4. Context generation for LLM augmentation
"""

import os
import csv
import json
import pickle
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

# Lazy imports for dependencies
try:
    import numpy as np
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import faiss
    # Suppress FAISS AVX512 warning - AVX2 works fine
    import logging
    faiss_logger = logging.getLogger('faiss.loader')
    faiss_logger.setLevel(logging.ERROR)
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Missing dependencies. Install with: pip install sentence-transformers pandas numpy scikit-learn faiss-cpu")
    print(f"Error: {e}")
    DEPENDENCIES_AVAILABLE = False
    # Set dummy imports to avoid immediate failure
    np = None
    pd = None
    SentenceTransformer = None
    cosine_similarity = None
    faiss = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Represents a document in the RAG system"""
    id: str
    content: str
    metadata: Dict[str, Any]
    source: str
    doc_type: str

@dataclass
class RetrievalResult:
    """Result from document retrieval"""
    document: Document
    score: float
    rank: int

class ShopfloorDocumentProcessor:
    """Processes CSV files and creates documents for RAG"""
    
    def __init__(self, data_dir: str = "data"):
        if not DEPENDENCIES_AVAILABLE:
            raise RuntimeError("Required dependencies not available. Please install: pip install sentence-transformers pandas numpy scikit-learn faiss-cpu")
        self.data_dir = Path(data_dir)
        self.documents = []
    
    def process_csv_files(self) -> List[Document]:
        """Process all CSV files in data directory"""
        documents = []
        
        # Process node files
        node_files = {
            'facilityzones': 'nodes_facilityzones.csv',
            'machines': 'nodes_machines.csv',
            'operators': 'nodes_operators.csv',
            'sensors': 'nodes_sensors.csv',
            'workorders': 'nodes_workorders.csv',
            'maintenancelogs': 'nodes_maintenancelogs.csv',
            'productionbatches': 'nodes_productionbatches.csv'
        }
        
        for node_type, filename in node_files.items():
            file_path = self.data_dir / filename
            if file_path.exists():
                docs = self._process_node_file(file_path, node_type)
                documents.extend(docs)
                logger.info(f"Processed {len(docs)} documents from {filename}")
        
        # Process relationship files
        rel_files = {
            'has_log': 'rels_has_log.csv',
            'has_work_order': 'rels_has_work_order.csv',
            'located_in': 'rels_located_in.csv',
            'monitored_by': 'rels_monitored_by.csv',
            'operated_by': 'rels_operated_by.csv',
            'part_of_batch': 'rels_part_of_batch.csv'
        }
        
        for rel_type, filename in rel_files.items():
            file_path = self.data_dir / filename
            if file_path.exists():
                docs = self._process_relationship_file(file_path, rel_type)
                documents.extend(docs)
                logger.info(f"Processed {len(docs)} relationship documents from {filename}")
        
        self.documents = documents
        return documents
    
    def _process_node_file(self, file_path: Path, node_type: str) -> List[Document]:
        """Process a node CSV file"""
        documents = []
        
        try:
            df = pd.read_csv(file_path)
            for idx, row in df.iterrows():
                # Create content from all columns
                content_parts = []
                metadata = {}
                
                for col, value in row.items():
                    if pd.notna(value):
                        content_parts.append(f"{col}: {value}")
                        metadata[col] = value
                
                content = f"{node_type.title()} Entity - " + " | ".join(content_parts)
                
                doc = Document(
                    id=f"{node_type}_{idx}",
                    content=content,
                    metadata=metadata,
                    source=str(file_path),
                    doc_type=f"entity_{node_type}"
                )
                documents.append(doc)
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
        
        return documents
    
    def _process_relationship_file(self, file_path: Path, rel_type: str) -> List[Document]:
        """Process a relationship CSV file"""
        documents = []
        
        try:
            df = pd.read_csv(file_path)
            for idx, row in df.iterrows():
                # Create content describing the relationship
                content_parts = []
                metadata = {}
                
                for col, value in row.items():
                    if pd.notna(value):
                        content_parts.append(f"{col}: {value}")
                        metadata[col] = value
                
                content = f"{rel_type.replace('_', ' ').title()} Relationship - " + " | ".join(content_parts)
                
                doc = Document(
                    id=f"{rel_type}_{idx}",
                    content=content,
                    metadata=metadata,
                    source=str(file_path),
                    doc_type=f"relationship_{rel_type}"
                )
                documents.append(doc)
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
        
        return documents

class VectorStore:
    """Vector store for semantic search"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not DEPENDENCIES_AVAILABLE:
            raise RuntimeError("Required dependencies not available. Please install: pip install sentence-transformers pandas numpy scikit-learn faiss-cpu")
        
        # Configure SSL settings for corporate networks BEFORE any network calls
        import os
        import ssl
        import urllib3
        
        # Disable SSL verification globally for corporate networks
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Set environment variables to disable SSL verification
        os.environ['CURL_CA_BUNDLE'] = ''
        os.environ['REQUESTS_CA_BUNDLE'] = ''
        os.environ['SSL_CERT_FILE'] = ''
        
        logger.info(f"Attempting to load SentenceTransformer model: {model_name}")
        logger.info("Note: SSL verification disabled for corporate network compatibility")
            
        try:
            # Try to load the model with SSL disabled
            self.model = SentenceTransformer(model_name)
            logger.info(f"✅ Successfully loaded SentenceTransformer model: {model_name}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to download model from Hugging Face: {e}")
            logger.info("💡 Switching to simple fallback embedding method...")
            
            # Use simple fallback that doesn't require model download
            self.model = None
            logger.info("✅ Using fallback embedding (TF-IDF-like) for RAG search")
        
        self.documents = []
        self.embeddings = None
        self.index = None
        self.is_built = False
    
    def _simple_embeddings(self, texts: List[str]):
        """Simple fallback embedding method using TF-IDF-like approach"""
        from collections import Counter
        import string
        
        # Simple tokenization and vectorization
        all_words = set()
        doc_words = []
        
        for text in texts:
            # Simple preprocessing
            words = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
            doc_words.append(words)
            all_words.update(words)
        
        vocab = list(all_words)
        vocab_size = len(vocab)
        word_to_idx = {word: i for i, word in enumerate(vocab)}
        
        # Create simple embeddings (like bag of words with some weighting)
        embeddings = np.zeros((len(texts), vocab_size))
        
        for i, words in enumerate(doc_words):
            word_counts = Counter(words)
            for word, count in word_counts.items():
                if word in word_to_idx:
                    embeddings[i, word_to_idx[word]] = count / len(words)  # Simple TF
        
        return embeddings.astype(np.float32)

    def build_index(self, documents: List[Document]):
        """Build vector index from documents"""
        self.documents = documents
        
        # Extract text content
        texts = [doc.content for doc in documents]
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} documents...")
        
        if self.model is not None:
            # Use sentence transformer
            self.embeddings = self.model.encode(texts, show_progress_bar=True)
        else:
            # Use simple fallback
            logger.info("Using simple fallback embedding method")
            self.embeddings = self._simple_embeddings(texts)
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        self.is_built = True
        logger.info(f"Vector index built with {len(documents)} documents")
    
    def search(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """Search for similar documents"""
        if not self.is_built:
            raise ValueError("Index not built. Call build_index first.")
        
        # Encode query
        if self.model is not None:
            query_embedding = self.model.encode([query])
            faiss.normalize_L2(query_embedding)
        else:
            # Use simple fallback for query embedding
            query_embedding = self._simple_embeddings([query])
            faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):  # Valid index
                result = RetrievalResult(
                    document=self.documents[idx],
                    score=float(score),
                    rank=rank + 1
                )
                results.append(result)
        
        return results
    
    def save(self, filepath: str):
        """Save vector store to disk"""
        data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'model_name': self.model.get_sentence_embedding_dimension()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        # Save FAISS index separately
        if self.index:
            faiss.write_index(self.index, filepath + '.faiss')
    
    def load(self, filepath: str):
        """Load vector store from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.documents = data['documents']
        self.embeddings = data['embeddings']
        
        # Load FAISS index
        index_path = filepath + '.faiss'
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            self.is_built = True

class RAGSystem:
    """Main RAG system coordinator"""
    
    def __init__(self, data_dir: str = "data", model_name: str = "all-MiniLM-L6-v2"):
        if not DEPENDENCIES_AVAILABLE:
            raise RuntimeError("Required dependencies not available. Please install: pip install sentence-transformers pandas numpy scikit-learn faiss-cpu")
        self.processor = ShopfloorDocumentProcessor(data_dir)
        self.vector_store = VectorStore(model_name)
        self.documents = []
    
    def initialize(self, force_rebuild: bool = False):
        """Initialize the RAG system"""
        cache_path = "rag_cache.pkl"
        
        if not force_rebuild and os.path.exists(cache_path):
            logger.info("Loading cached RAG system...")
            try:
                self.vector_store.load(cache_path)
                self.documents = self.vector_store.documents
                logger.info(f"Loaded {len(self.documents)} documents from cache")
                return
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Rebuilding...")
        
        # Process documents
        logger.info("Processing shopfloor documents...")
        self.documents = self.processor.process_csv_files()
        
        # Build vector index
        logger.info("Building vector index...")
        self.vector_store.build_index(self.documents)
        
        # Save cache
        logger.info("Saving cache...")
        self.vector_store.save(cache_path)
        
        logger.info(f"RAG system initialized with {len(self.documents)} documents")
    
    def query(self, question: str, k: int = 5, include_metadata: bool = True) -> Dict[str, Any]:
        """Query the RAG system"""
        if not self.vector_store.is_built:
            raise ValueError("RAG system not initialized. Call initialize() first.")
        
        # Retrieve relevant documents
        results = self.vector_store.search(question, k)
        
        # Format response
        context_docs = []
        for result in results:
            doc_info = {
                'content': result.document.content,
                'score': result.score,
                'rank': result.rank,
                'doc_type': result.document.doc_type,
                'source': result.document.source
            }
            
            if include_metadata:
                doc_info['metadata'] = result.document.metadata
            
            context_docs.append(doc_info)
        
        # Create context for LLM
        context = self._create_context(results)
        
        return {
            'query': question,
            'context': context,
            'retrieved_documents': context_docs,
            'num_results': len(results)
        }
    
    def _create_context(self, results: List[RetrievalResult]) -> str:
        """Create formatted context for LLM"""
        if not results:
            return "No relevant information found."
        
        context_parts = ["Based on the shopfloor data, here's the relevant information:"]
        
        for i, result in enumerate(results, 1):
            context_parts.append(f"\n{i}. {result.document.content}")
            if result.document.metadata:
                # Add key metadata
                key_fields = ['id', 'name', 'type', 'status', 'timestamp']
                metadata_parts = []
                for field in key_fields:
                    if field in result.document.metadata:
                        metadata_parts.append(f"{field}: {result.document.metadata[field]}")
                if metadata_parts:
                    context_parts.append(f"   ({', '.join(metadata_parts)})")
        
        return "\n".join(context_parts)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        doc_types = {}
        sources = {}
        
        for doc in self.documents:
            doc_types[doc.doc_type] = doc_types.get(doc.doc_type, 0) + 1
            sources[doc.source] = sources.get(doc.source, 0) + 1
        
        return {
            'total_documents': len(self.documents),
            'document_types': doc_types,
            'data_sources': sources,
            'vector_store_built': self.vector_store.is_built
        }

# CLI interface for testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python rag_system.py <query>")
        print("Example: python rag_system.py 'What machines have high vibration?'")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    
    # Initialize RAG system
    rag = RAGSystem()
    rag.initialize()
    
    # Query
    result = rag.query(query)
    
    print(f"\nQuery: {result['query']}")
    print(f"\nContext:\n{result['context']}")
    print(f"\nFound {result['num_results']} relevant documents")
    
    # Print statistics
    stats = rag.get_statistics()
    print(f"\nSystem Statistics:")
    print(json.dumps(stats, indent=2))


class SimpleRAGAgent:
    """Simplified RAG agent for easy integration"""
    
    def __init__(self):
        self.rag_system = RAGSystem()
        self.initialized = False
        self.stats = {
            'queries_processed': 0,
            'total_execution_time': 0,
            'average_execution_time': 0,
            'errors': 0
        }
    
    async def initialize(self):
        """Initialize the RAG system"""
        try:
            self.rag_system.initialize()
            # Check if initialization was successful by checking if we have documents
            success = len(self.rag_system.documents) > 0 if hasattr(self.rag_system, 'documents') else True
            self.initialized = success
            return success
        except Exception as e:
            logger.error(f"Failed to initialize SimpleRAGAgent: {e}")
            return False
    
    async def search_documents(self, query: str, k: int = 5, include_metadata: bool = True):
        """Search for documents using the RAG system"""
        import time
        start_time = time.time()
        
        try:
            if not self.initialized:
                return {
                    'success': False,
                    'error': 'RAG system not initialized',
                    'query': query,
                    'execution_time': 0
                }
            
            result = self.rag_system.query(query, k=k)
            execution_time = time.time() - start_time
            
            self._update_stats(execution_time)
            
            return {
                'success': True,
                'query': result['query'],
                'num_results': result['num_results'],
                'retrieved_documents': result.get('documents', []),
                'execution_time': execution_time
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_stats(execution_time, error=True)
            return {
                'success': False,
                'error': str(e),
                'query': query,
                'execution_time': execution_time
            }
    
    async def get_context(self, query: str, k: int = 3, max_context_length: int = 2000):
        """Get context for a query
        
        Args:
            query: The search query
            k: Number of documents to retrieve (default: 3)
            max_context_length: Maximum length of context to return
        """
        result = await self.search_documents(query, k=k)
        if result['success']:
            # Build context from retrieved documents
            context_parts = []
            for doc in result['retrieved_documents']:
                context_parts.append(doc.get('content', ''))
            
            context = '\n\n'.join(context_parts)
            if len(context) > max_context_length:
                context = context[:max_context_length] + "..."
            
            return {
                'success': True,
                'context': context,
                'num_documents': result['num_results']
            }
        else:
            return result
    
    def _update_stats(self, execution_time: float, error: bool = False):
        """Update performance statistics"""
        self.stats['queries_processed'] += 1
        self.stats['total_execution_time'] += execution_time
        self.stats['average_execution_time'] = self.stats['total_execution_time'] / self.stats['queries_processed']
        
        if error:
            self.stats['errors'] += 1
    
    def get_statistics(self):
        """Get performance statistics"""
        return self.stats.copy()
    
    def get_status(self):
        """Get agent status for compatibility with enhanced main agent"""
        return {
            'active': self.initialized,
            'type': 'SimpleRAGAgent',
            'stats': self.stats,
            'rag_system_initialized': self.rag_system.initialized if hasattr(self.rag_system, 'initialized') else False
        }
    
    async def cleanup(self):
        """Cleanup method for compatibility"""
        logger.info("SimpleRAGAgent cleanup completed")
        pass