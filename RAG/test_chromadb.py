import chromadb

# Tạo database
client = chromadb.PersistentClient(path="./chroma_db")

# Tạo collection (nơi lưu các vector embedding)
collection = client.get_or_create_collection(name="my_vectors")

# Lưu vector vào database với metadata
collection.add(
    ids=["1", "2", "3"],
    embeddings=[
        [0.1, 0.2, 0.3],  # Vector 1
        [0.4, 0.5, 0.6],  # Vector 2
        [0.7, 0.8, 0.9]   # Vector 3
    ],
    metadatas=[
        {"text": "Xin chào!", "source": "user_input"},
        {"text": "Học máy là gì?", "source": "Wikipedia"},
        {"text": "AI có thể làm gì?", "source": "Book"}
    ]
)

# Truy vấn vector gần nhất
query_vector = [0.1, 0.2, 0.3]  # Một vector query
results = collection.query(
    query_embeddings=[query_vector],
    n_results=1
)
print(results)  # Kết quả trả về ID, vector, metadata tương ứng
