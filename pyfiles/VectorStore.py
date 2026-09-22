import lancedb


def _table_names(db) -> list[str]:
    listed = db.list_tables()
    return listed.tables if hasattr(listed, "tables") else list(listed)


def vectore_store(records, output_dir: str, table_name: str, embedding_dim: int = 1024):
    del embedding_dim  # kept for callers; schema is inferred from record vectors
    db = lancedb.connect(output_dir)

    if table_name not in _table_names(db):
        # Let Lance infer the vector column so Linux CI builds treat it as a vector.
        table = db.create_table(table_name, data=records)
    else:
        table = db.open_table(table_name)
        table.add(records) 

    return table