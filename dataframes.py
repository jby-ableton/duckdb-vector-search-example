import duckdb
import numpy as np
import pandas as pd

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

# Prepare data using pandas DataFrame
data = pd.DataFrame({
    'id': range(1, 6),
    'description': ['Red apple', 'Green apple', 'Blue berry', 'Yellow banana', 'Purple grape'],
    'vec': [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0]
    ]
})

# Insert data into DuckDB table
conn.execute("INSERT INTO embeddings SELECT * FROM data")

# Create an HNSW index on the vector column
conn.execute("CREATE INDEX embedding_idx ON embeddings USING HNSW (vec)")

# Perform a similarity search
query_vector = [0.8, 0.1, 0.1]
result = conn.execute("""
    SELECT id, description, array_distance(vec, ?::FLOAT[3]) as distance
    FROM embeddings
    ORDER BY array_distance(vec, ?::FLOAT[3])
    LIMIT 3
""", [query_vector, query_vector]).fetchall()

print("Top 3 most similar items:")
for row in result:
    print(f"ID: {row[0]}, Description: {row[1]}, Distance: {row[2]:.4f}")

# Close the connection
conn.close()