import duckdb
import csv
import os

DATA_FILE = 'embeddings.csv'

def create_sample_data():
    if not os.path.exists(DATA_FILE):
        print(f"Creating sample data file: {DATA_FILE}")
        with open(DATA_FILE, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['id', 'description', 'vec'])
            writer.writerows([
                [1, 'Red apple', '[1.0,0.0,0.0]'],
                [2, 'Green apple', '[0.0,1.0,0.0]'],
                [3, 'Blue berry', '[0.0,0.0,1.0]'],
                [4, 'Yellow banana', '[1.0,1.0,0.0]'],
                [5, 'Purple grape', '[1.0,0.0,1.0]']
            ])
        print(f"Sample data file created: {DATA_FILE}")
    else:
        print(f"Sample data file already exists: {DATA_FILE}")

def initialize_database():
    conn = duckdb.connect(':memory:')
    
    # Install and load the vss extension
    conn.execute("INSTALL vss")
    conn.execute("LOAD vss")
    
    # Create the table from the CSV file with explicit column specifications
    conn.execute(f"""
        CREATE TABLE embeddings AS 
        SELECT 
            CAST(id AS INTEGER) AS id,
            description,
            CAST(vec AS FLOAT[3]) AS vec
        FROM read_csv('{DATA_FILE}', header=true, columns={{
            'id': 'VARCHAR',
            'description': 'VARCHAR',
            'vec': 'VARCHAR'
        }})
    """)
    
    # Create an HNSW index on the vector column
    conn.execute("CREATE INDEX embedding_idx ON embeddings USING HNSW (vec)")
    print("Table created from CSV file and HNSW index built.")
    
    return conn

def perform_similarity_search(conn, query_vector):
    result = conn.execute("""
        SELECT id, description, array_distance(vec, ?::FLOAT[3]) as distance
        FROM embeddings
        ORDER BY array_distance(vec, ?::FLOAT[3])
        LIMIT 3
    """, [query_vector, query_vector]).fetchall()
    
    print("\nTop 3 most similar items:")
    for row in result:
        print(f"ID: {row[0]}, Description: {row[1]}, Distance: {row[2]:.4f}")

def main():
    create_sample_data()
    conn = initialize_database()
    
    query_vector = [0.8, 0.1, 0.1]
    perform_similarity_search(conn, query_vector)
    
    conn.close()

if __name__ == "__main__":
    main()