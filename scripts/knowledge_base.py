import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def search_knowledge(query, pdf_path, db_save_dir="data/vector_db", top_k=3):
    """
    诊疗知识库检索函数
    输入：问题、PDF指南路径、向量库保存目录、返回最相关的片段数
    输出：列表，每个元素是一段相关的指南原文
    """
    # 初始化嵌入模型
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    # 判断向量库是否已存在
    if not os.path.exists(db_save_dir) or not os.listdir(db_save_dir):
        # 加载PDF并切分
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", " ", ""]
        )
        splits = text_splitter.split_documents(documents)
        
        # 构建向量库并保存
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=db_save_dir
        )
        vectorstore.persist()
    else:
        # 直接加载已有向量库
        vectorstore = Chroma(
            persist_directory=db_save_dir,
            embedding_function=embeddings
        )
    
    # 检索并返回纯文本
    results = vectorstore.similarity_search(query, k=top_k)
    return [doc.page_content for doc in results]


if __name__ == "__main__":
    result = search_knowledge(
        query="肺炎的诊断标准是什么",
        pdf_path=r"D:\med_img_project\data\cap_guide_2018.pdf"
    )
    for i, text in enumerate(result):
        print(f"--- 相关片段 {i+1} ---")
        print(text[:300] + "\n")