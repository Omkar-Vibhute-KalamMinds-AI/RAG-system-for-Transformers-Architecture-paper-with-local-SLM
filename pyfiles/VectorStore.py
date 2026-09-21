import lancedb  
import pyarrow as pa
import numpy as np 

def _table_names(db) -> list[str]:
    listed = db.list_tables()
    return listed.tables if hasattr(listed, "tables") else list(listed)


def vectore_store(records, output_dir: str, table_name: str, embedding_dim: int = 1024):
    
    db = lancedb.connect(output_dir)

    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("client_id", pa.string()), 
        pa.field("doc_id", pa.string()),
        pa.field("source_file", pa.string()),
        pa.field("chunk_index", pa.int64()),
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), embedding_dim)),
    ])

    if table_name not in _table_names(db):
        table = db.create_table(table_name, schema=schema, data=records)
        
    else:
        table = db.open_table(table_name)
        table.add(records)   

    return table
#----------------------------------------------

"""
source_file = r'D:\RAG Learn\Data\Attention is that all you need.pdf'
client_id = 'Advanced_RAG_Method'
doc_id =  'Attention_is_all_you_need'

records = []
for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
    records.append({
        "id": f"{doc_id}_chunk_{i:04d}",
        "client_id": client_id,
        "doc_id": doc_id,
        "source_file": 'Attention is that all you need.pdf',
        "chunk_index": i,
        "text": chunk,  
        "vector": emb.astype(np.float32).tolist(), 
    })
"""

#transformer_table = vectore_store(records=records , output_dir = r'D:\RAG Learn\lance_db\Attention_paper_db', table_name= 'transformer_table', embedding_dim = 1024) 

#transformers_table.create_index( metric="cosine",
 #   num_partitions= 20,
#    num_sub_vectors = 64, 
#    vector_column_name="vector"
#)