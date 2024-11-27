from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastembed.embedding import TextEmbedding
from fastembed.sparse.bm25 import Bm25
from qdrant_client import QdrantClient, models
import os
from langchain.document_loaders import TextLoader

# Initialize global variables
collection_name = "test_collection"
client = QdrantClient(
    url="http://localhost:6333", 
    api_key="##",
)
dense_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
sparse_model = Bm25("Qdrant/bm25")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=60,
    length_function=len,
)

# Create a Qdrant collection with both dense and sparse vector configurations.
def create_collection():
    client.create_collection(
        collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=384,
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        }
    )

 # Read all txt files from a directory and return their contents as documents. 
def read_documents_from_directory(directory):
    all_documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            print(f"Processing file: {filename}")
            try:
                loader = TextLoader(file_path)
                documents = loader.load()
                all_documents.extend(documents)
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
    return all_documents



# Process all txt documents in  directory and index them.
def process_and_index_directory(directory):
    
    # Read all documents
    documents = read_documents_from_directory(directory)
    
    # Process each document
    all_points = []
    current_id = 0
    
    for doc in documents:
        # Process the document content
        chunks = text_splitter.split_text(doc.page_content)
        
        for chunk in chunks:
            dense_embedding = list(dense_model.passage_embed([chunk]))[0]
            sparse_embedding = list(sparse_model.passage_embed([chunk]))[0]
            
            point = {
                "id": current_id,
                "vector": {
                    "dense": dense_embedding.tolist(),
                },
                "payload": {
                    "text": chunk,
                    "original_position": current_id,
                    "source": doc.metadata.get('source', 'unknown')  # Include source file information
                }
            }
            
            if sparse_embedding is not None:
                point["vector"]["sparse"] = sparse_embedding.as_object()
            
            all_points.append(point)
            current_id += 1
    
    # Batch upload points to Qdrant
    if all_points:
        client.upsert(
            collection_name=collection_name,
            points=[models.PointStruct(**point) for point in all_points]
        )
        print(f"Successfully indexed {len(all_points)} chunks from {len(documents)} documents")


# Perform hybrid search using both dense and sparse vectors.


def hybrid_search(query, limit=5):
    dense_query = list(dense_model.passage_embed([query]))[0]
    sparse_query = list(sparse_model.passage_embed([query]))[0]
    
    try:
        prefetch = [
            models.Prefetch(
                query=dense_query.tolist(),
                using="dense",
                limit=20,
            ),
            models.Prefetch(
                query=models.SparseVector(**sparse_query.as_object()),
                using="sparse",
                limit=20,
            )
        ]
        
        results = client.query_points(
            collection_name=collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(
                fusion=models.Fusion.RRF,
            ),
            limit=limit,
            with_payload=True
        )
        
        formatted_results = []
        if hasattr(results, 'points'):
            for point in results.points:
                formatted_results.append({
                    'text': point.payload['text'],
                    'score': point.score,
                    'position': point.payload['original_position'],
                    'source': point.payload.get('source', 'unknown')
                })
        
        return formatted_results
        
    except Exception as e:
        print(f"Search error: {str(e)}")
        return []

def main():
    # Create collection (only needed once)
    try:
        create_collection()
    except Exception as e:
        print(f"Collection might already exist: {e}")
    
    # Process and index all documents in a directory
    documents_directory = "faq"
    process_and_index_directory(documents_directory)
    
    # Search example
    results = hybrid_search("why are my requests are very slow", limit=3)
    
    # Print results
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Source: {result['source']}")
        print(f"Text: {result['text']}")
        print(f"Score: {result['score']}")
        print(f"Original Position: {result['position']}")

if __name__ == "__main__":
    main()
