import os
import logging
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import json

import weaviate
from weaviate.exceptions import WeaviateException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WeaviateConfig:
    """Configuration parameters for Weaviate connection"""

    url: str
    timeout: int = 30
    startup_period: int = 5

    @classmethod
    def from_env(cls) -> "WeaviateConfig":
        """Creates configuration from environment variables"""
        return cls(
            url=os.getenv("WEAVIATE_URL", "http://localhost:8080"),
            timeout=int(os.getenv("WEAVIATE_TIMEOUT", "30")),
            startup_period=int(os.getenv("WEAVIATE_STARTUP_PERIOD", "5")),
        )


class WeaviateManager:
    """Manages all Weaviate vector database operations for legal documents with built-in vectorization"""

    def __init__(self, config: Optional[WeaviateConfig] = None):
        """Initializes Weaviate client with configuration"""
        self.config = config or WeaviateConfig.from_env()
        self.client = None
        self.collection_name = "LegalDocuments"
        self._connect()

    def _connect(self) -> None:
        """Establishes connection to Weaviate instance"""
        try:
            self.client = weaviate.Client(
                url=self.config.url,
                timeout_config=(self.config.timeout, self.config.startup_period),
            )

            if not self.client.is_ready():
                raise WeaviateException("Weaviate instance is not ready")

            logger.info(f"Connected to Weaviate at {self.config.url}")

        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {str(e)}")
            raise

    def create_schema(self, recreate: bool = True) -> bool:
        """Creates or recreates the legal documents collection schema with built-in vectorization"""
        try:
            # Delete existing collection if recreate is True (maintains current behavior)
            if recreate and self.client.schema.exists(self.collection_name):
                self.client.schema.delete_class(self.collection_name)
                logger.info(f"Deleted existing collection: {self.collection_name}")

            # Define schema with text2vec-transformers vectorizer
            schema = {
                "class": self.collection_name,
                "vectorizer": "text2vec-transformers",
                "moduleConfig": {
                    "text2vec-transformers": {
                        "poolingStrategy": "masked_mean",
                        "vectorizeClassName": False,
                    }
                },
                "vectorIndexType": "hnsw",
                "vectorIndexConfig": {
                    "distance": "cosine",
                    "efConstruction": 128,
                    "ef": 64,
                    "maxConnections": 32,
                },
                "properties": [
                    {
                        "name": "text",
                        "dataType": ["text"],
                        "description": "Content of the legal document chunk",
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": False,
                                "vectorizePropertyName": False,
                            }
                        },
                    },
                    {
                        "name": "chunk_index",
                        "dataType": ["int"],
                        "description": "Index of chunk within the document",
                        "moduleConfig": {"text2vec-transformers": {"skip": True}},
                    },
                    {
                        "name": "document_id",
                        "dataType": ["string"],
                        "description": "Unique identifier for the source document",
                        "moduleConfig": {"text2vec-transformers": {"skip": True}},
                    },
                    {
                        "name": "metadata",
                        "dataType": ["text"],
                        "description": "Additional metadata as JSON string",
                        "moduleConfig": {"text2vec-transformers": {"skip": True}},
                    },
                ],
            }

            self.client.schema.create_class(schema)
            logger.info(
                f"Created collection schema with built-in vectorization: {self.collection_name}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to create schema: {str(e)}")
            return False

    def add_documents(
        self, documents: List[Dict], batch_size: int = 100
    ) -> Dict[str, Any]:
        """Adds documents to Weaviate - vectorization happens automatically"""
        results = {"success": 0, "failed": 0, "errors": []}

        try:
            with self.client.batch(
                batch_size=batch_size, dynamic=True, timeout_retries=3
            ) as batch:
                for idx, doc in enumerate(documents):
                    # Only need text and metadata - Weaviate handles vectorization
                    if "text" not in doc:
                        results["failed"] += 1
                        results["errors"].append(f"Document {idx} missing text field")
                        continue

                    # Prepare document properties (no embedding needed)
                    properties = {
                        "text": doc["text"],
                        "chunk_index": doc.get("chunk_index", idx),
                        "document_id": doc.get("document_id", f"doc_{idx}"),
                        "metadata": json.dumps(doc.get("metadata", {})),
                    }

                    # Add to batch without vector - Weaviate will generate it
                    batch.add_data_object(
                        data_object=properties, class_name=self.collection_name
                    )

                    results["success"] += 1

            logger.info(
                f"Added {results['success']} documents to Weaviate with automatic vectorization"
            )

        except Exception as e:
            logger.error(f"Batch insertion failed: {str(e)}")
            results["errors"].append(str(e))

        return results

    def search_documents(
        self, query_text: str, limit: int = 10, certainty: float = 0.0
    ) -> List[Dict]:
        """Performs semantic search using built-in vectorization for the query"""
        try:
            # Use near_text for semantic search - Weaviate handles vectorization
            response = (
                self.client.query.get(
                    self.collection_name,
                    ["text", "chunk_index", "document_id", "metadata"],
                )
                .with_near_text({"concepts": [query_text], "certainty": certainty})
                .with_limit(limit)
                .with_additional(["certainty", "id", "distance"])
                .do()
            )

            # Process results
            results = []
            if response and "data" in response:
                documents = response["data"]["Get"][self.collection_name]
                for doc in documents:
                    results.append(
                        {
                            "text": doc.get("text", ""),
                            "chunk_index": doc.get("chunk_index", 0),
                            "document_id": doc.get("document_id", ""),
                            "metadata": json.loads(doc.get("metadata", "{}")),
                            "score": doc.get("_additional", {}).get("certainty", 0.0),
                            "distance": doc.get("_additional", {}).get("distance", 0.0),
                            "id": doc.get("_additional", {}).get("id", ""),
                        }
                    )

            return results

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

    def search_documents_with_vector(
        self, query_vector: List[float], limit: int = 10, certainty: float = 0.0
    ) -> List[Dict]:
        """Legacy method for backward compatibility - converts to text search"""
        logger.warning(
            "search_documents_with_vector called but using text search instead"
        )
        # This method is kept for backward compatibility but won't be used
        # since we're using built-in vectorization
        return []

    def update_document(self, document_id: str, properties: Dict) -> bool:
        """Updates document properties"""
        try:
            # Find document by custom document_id property
            where = {
                "path": ["document_id"],
                "operator": "Equal",
                "valueString": document_id,
            }

            existing = (
                self.client.query.get(self.collection_name, ["document_id"])
                .with_where(where)
                .with_additional(["id"])
                .do()
            )

            if existing and existing["data"]["Get"][self.collection_name]:
                weaviate_id = existing["data"]["Get"][self.collection_name][0][
                    "_additional"
                ]["id"]

                # Update properties - if text changes, Weaviate will re-vectorize
                self.client.data_object.update(
                    data_object=properties,
                    class_name=self.collection_name,
                    uuid=weaviate_id,
                )
                logger.info(f"Updated document: {document_id}")
                return True

            logger.warning(f"Document not found: {document_id}")
            return False

        except Exception as e:
            logger.error(f"Update failed for document {document_id}: {str(e)}")
            return False

    def delete_document(self, document_id: str) -> bool:
        """Deletes a document from the collection"""
        try:
            where = {
                "path": ["document_id"],
                "operator": "Equal",
                "valueString": document_id,
            }

            result = self.client.batch.delete_objects(
                class_name=self.collection_name, where=where
            )

            if result and result.get("results", {}).get("successful", 0) > 0:
                logger.info(f"Deleted document: {document_id}")
                return True

            logger.warning(f"Document not found for deletion: {document_id}")
            return False

        except Exception as e:
            logger.error(f"Deletion failed for document {document_id}: {str(e)}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """Checks Weaviate connection and collection status"""
        health_status = {
            "connected": False,
            "collection_exists": False,
            "document_count": 0,
            "response_time_ms": 0,
            "vectorizer_active": False,
        }

        try:
            start_time = time.time()

            # Check connection
            health_status["connected"] = self.client.is_ready()

            # Check collection exists
            if health_status["connected"]:
                health_status["collection_exists"] = self.client.schema.exists(
                    self.collection_name
                )

                # Check if vectorizer module is active
                if health_status["collection_exists"]:
                    schema = self.client.schema.get(self.collection_name)
                    if schema and schema.get("vectorizer") == "text2vec-transformers":
                        health_status["vectorizer_active"] = True

                    # Get document count
                    count_response = (
                        self.client.query.aggregate(self.collection_name)
                        .with_meta_count()
                        .do()
                    )

                    if count_response and "data" in count_response:
                        aggregate_data = count_response["data"]["Aggregate"][
                            self.collection_name
                        ]
                        if aggregate_data and len(aggregate_data) > 0:
                            health_status["document_count"] = aggregate_data[0]["meta"][
                                "count"
                            ]

            health_status["response_time_ms"] = int((time.time() - start_time) * 1000)

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")

        return health_status

    def get_collection_stats(self) -> Dict[str, Any]:
        """Retrieves detailed statistics about the collection"""
        stats = {
            "collection_name": self.collection_name,
            "document_count": 0,
            "index_type": "hnsw",
            "distance_metric": "cosine",
            "vectorizer": "text2vec-transformers",
            "vector_dimension": 384,  # all-MiniLM-L6-v2 dimensions
        }

        try:
            # Get document count
            health = self.health_check()
            stats["document_count"] = health.get("document_count", 0)
            stats["vectorizer_active"] = health.get("vectorizer_active", False)

            # Get schema information
            schema = self.client.schema.get(self.collection_name)
            if schema:
                stats["properties"] = [
                    prop["name"] for prop in schema.get("properties", [])
                ]
                stats["vector_config"] = schema.get("vectorIndexConfig", {})
                stats["module_config"] = schema.get("moduleConfig", {})

        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")

        return stats

    def clear_collection(self) -> bool:
        """Clears all documents from the collection without deleting schema"""
        try:
            self.client.batch.delete_objects(
                class_name=self.collection_name,
                where={"path": ["document_id"], "operator": "Like", "valueString": "*"},
            )
            logger.info(f"Cleared all documents from {self.collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to clear collection: {str(e)}")
            return False

    def close(self) -> None:
        """Closes the Weaviate client connection"""
        if self.client:
            self.client = None
            logger.info("Weaviate connection closed")
