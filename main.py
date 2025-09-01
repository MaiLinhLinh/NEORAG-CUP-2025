from docx import Document
from embeddings import Embeddings
from vector_db import VectorDatabase
import pandas as pd
import openai
import os
from sentence_transformers import SentenceTransformer


# Hàm xử lí document
import re
def is_header(t:str)->bool:
    s = (t or "").rstrip()
    return bool(s) and s.endswith(":")

def normalize_header(t: str) -> str:
    # bỏ dấu “:” và khoảng trắng ở cuối
    return re.sub(r'[\s:]+$', '', (t or '').strip())

def records_from_docx(doc):
    
    """
    Trả về list các bản ghi đã 'gắn tiêu đề' cho từng dòng nội dung.
    Mỗi bản ghi có:
      - section_title: tiêu đề (hoặc None nếu không có)
      - text: nội dung gốc của dòng
      - information: chuỗi cuối cùng để embed (Tiêu đề: Nội dung) hoặc chỉ 'Nội dung' nếu không có tiêu đề
    """

    out = []
    current_title = None

    for p in doc.paragraphs:
        line = (p.text or "").strip()
        if not line:
            continue
        if is_header(line):
            current_title = line
            out.append({
                "section_title": current_title,
                "text": line,
                "information": line
            })
            continue
        if current_title:
            info = f"{current_title} {line}"
            out.append({
                "section_title": current_title,
                "text": line,
                "information": info
            })
        else:
            out.append({
                "section_title": current_title,
                "text": line,
                "information": line
            })
    return out


doc = Document("CLB_PROPTIT.docx")


vector_db = VectorDatabase(db_type= "mongodb") # baseline model: "mongodb"

# Sửa chỗ FIX_ME để dùng embedding model mà các em muốn, hoặc các em có tự thêm embedding model trong lớp Embeddings

embedding = Embeddings(model_name= "text-embedding-3-large", type= "openai") # baseline model: "text-embedding-3-large", "openai"



# TODO: Embedding từng document trong file CLB_PROPTIT.docx và lưu vào DB. 
# Code dưới là sử dụng mongodb, các em có thể tự sửa lại cho phù hợp với DB mà mình đang dùng
#--------------------Code Lưu Embedding Document vào DB--------------------------
cnt = 1
if vector_db.count_documents("information") == 0:
    records = records_from_docx(doc)
    for i, rec in enumerate(records, 1):
        info_text = rec["information"]
        embedding_vector = embedding.encode(info_text)
        # Lưu vào cơ sở dữ liệu
        vector_db.insert_document(
            collection_name="information",
            document={
                "title": f"Document {cnt}",
                "section_title": rec["section_title"],
                "information": info_text,
                "embedding": embedding_vector
            }
         )
        cnt += 1
else:
    print("Documents already exist in the database. Skipping insertion.")
#------------------------------------------------------------------------------------

# Các em có thể import từng hàm một để check kết quả, trick là nên chạy trên data nhỏ thôi để xem hàm có chạy đúng hay ko rồi mới chạy trên toàn bộ data
'''
from metrics_rag import calculate_metrics_retrieval, calculate_metrics_llm_answer

df_retrieval_metrics = calculate_metrics_retrieval("CLB_PROPTIT.csv", "train_data_proptit.xlsx", embedding, vector_db, True) # đặt là True nếu là tập train, False là tập test
#df_llm_metrics = calculate_metrics_llm_answer("CLB_PROPTIT.csv", "train_data_proptit.xlsx", embedding, vector_db, True) # đặt là True nếu là tập train, False là tập test
print(df_retrieval_metrics.head())
#print(df_llm_metrics.head())
'''
'''
from dotenv import load_dotenv

load_dotenv()
print("check log")

'''


from metrics_rag import ndcg_k
ndcg_k_value = ndcg_k("CLB_PROPTIT.csv", "test_data_proptit.xlsx",embedding, vector_db, k=5)
print(f"ndcg_k@3: {ndcg_k_value:.4f}")


'''
from metrics_rag import noise_sensitivity_k
noise_sensitivity_k_value = noise_sensitivity_k("CLB_PROPTIT.csv","train_data_proptit.xlsx", embedding, vector_db, k= 7)
print(f"response_relevancy@3: {noise_sensitivity_k_value:.4f}")
'''