import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings

def ingest_all_lessons(base_path="data"):
    print("📂 Loading lesson files...")
    all_docs = []

    for class_dir in os.listdir(base_path):
        class_path = os.path.join(base_path, class_dir)
        if not os.path.isdir(class_path):
            continue

        for subject_dir in os.listdir(class_path):
            subject_path = os.path.join(class_path, subject_dir)
            if not os.path.isdir(subject_path):
                continue

            for file in os.listdir(subject_path):
                if not file.endswith(".txt"):
                    continue

                lesson_path = os.path.join(subject_path, file)
                loader = TextLoader(lesson_path, encoding="utf-8")
                docs = loader.load()

                # Add metadata
                for doc in docs:
                    doc.metadata["class"] = class_dir
                    doc.metadata["subject"] = subject_dir
                    doc.metadata["lesson"] = file.replace(".txt", "")
                    all_docs.append(doc)

    print(f"✅ Loaded {len(all_docs)} documents.")

    print("🧩 Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(all_docs)

    print("🔎 Creating embeddings...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("📦 Building vector store...")
    db = FAISS.from_documents(split_docs, embedding_model)

    vectorstore_path = os.path.join(os.path.dirname(__file__), "./vectorstore")
    db.save_local(vectorstore_path)
    print("✅ Vectorstore saved successfully at", vectorstore_path)

if __name__ == "__main__":
    ingest_all_lessons()
