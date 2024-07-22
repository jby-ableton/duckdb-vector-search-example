import duckdb

# Connect to an in-memory DuckDB database
conn = duckdb.connect(':memory:')

# Install and load the vss extension
conn.execute("INSTALL vss")
conn.execute("LOAD vss")

# Create a table with vector embeddings
conn.execute("""
    CREATE TABLE embeddings (
        id INTEGER,
        description VARCHAR,
        vec FLOAT[3]
    )
""")

# Insert some sample data
conn.execute("""
    INSERT INTO embeddings VALUES
    (1, 'Red apple', [1.0, 0.0, 0.0]),
    (2, 'Green apple', [0.0, 1.0, 0.0]),
    (3, 'Blue berry', [0.0, 0.0, 1.0]),
    (4, 'Yellow banana', [1.0, 1.0, 0.0]),
    (5, 'Purple grape', [1.0, 0.0, 1.0])
""")

# Create an HNSW index on the vector column
conn.execute("CREATE INDEX embedding_idx ON embeddings USING HNSW (vec)")

# Perform a similarity search
query_vector = [0.8, 0.1, 0.1]
result = conn.execute(f"""
    SELECT id, description, array_distance(vec, {query_vector}::FLOAT[3]) as distance
    FROM embeddings
    ORDER BY array_distance(vec, {query_vector}::FLOAT[3])
    LIMIT 3
""").fetchall()

print("Top 3 most similar items:")
for row in result:
    print(f"ID: {row[0]}, Description: {row[1]}, Distance: {row[2]:.4f}")

# Close the connection
conn.close()
