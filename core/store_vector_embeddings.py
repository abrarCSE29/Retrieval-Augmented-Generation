from vector_db.qdrant import Qdrant_VDB

vdb = Qdrant_VDB(
    collection_name="test_collection_384",
    vector_size=384
)


def store_embedding_vector(embedding_vec : list[list[float]], document_id : str):
    vdb.store_document_embeddings(
        document_id=document_id,
        embeddings=embedding_vec,
    )