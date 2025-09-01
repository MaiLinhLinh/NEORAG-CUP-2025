import streamlit as st
from embeddings import Embeddings
from vector_db import VectorDatabase
import pandas as pd
import os
from docx import Document
import numpy as np
import time
from functools import lru_cache
import openai
from dotenv import load_dotenv
from semantic_router.route import Route
from semantic_router.router import SemanticRouter
from semantic_router.samples import info_CLBSample
from semantic_router.samples import chitchatSample
from collections import defaultdict
from reflection import Reflection

load_dotenv()

@st.cache_resource
def _get_reranker():
    from rerank import Reranker           # import trong hàm để tránh import vòng & nặng
    return Reranker()                     # chỉ tạo 1 lần duy nhất

# Khởi tạo session_state để lưu hội thoại

def init_session():

    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role":"system",
                "content": """Bạn là một trợ lý AI chuyên cung cấp thông tin về Câu lạc bộ Lập trình ProPTIT.
                Bạn sẽ nhận được dữ liệu ngữ cảnh (context) từ một hệ thống Retrieval-Augmented Generation (RAG) chứa các thông tin chính xác về CLB.

                Nguyên tắc trả lời:
                1. Chỉ sử dụng thông tin từ context được cung cấp để trả lời. Context sẽ được cung cấp ở đầu mỗi query của người dùng. Câu hỏi của người dùng nằm ở đầu. 
                2. Nếu người dùng hỏi câu hỏi không liên quan đến CLB ProPTIT, hãy trả lời như bình thường, nhưng không sử dụng thông tin từ context.
                3. Trình bày câu trả lời rõ ràng, dễ hiểu. Có thể sử dụng emoij icon khi cần.
                4. Tuyệt đối không suy đoán hoặc bịa thông tin.
                5. Giữ phong cách trả lời thân thiện, chuyên nghiệp và nhất quán.
                6. Trong context có thể chứa nhiều thông tin khác nhau, hãy tập trung vào câu hỏi của người dùng để trả lời chính xác nhất.

                Nhiệm vụ của bạn:
                - Trả lời các câu hỏi về CLB Lập trình ProPTIT: lịch sử, thành viên, hoạt động, sự kiện, dự án, nội quy, thành viên tiêu biểu, và các thông tin liên quan khác.
                - Nếu câu hỏi không liên quan đến CLB ProPTIT thì trong câu trả lời không bao gồm bất cứ từ ngữ, thông tin về CLB.
                """
            }
        ]
    if "history" not in st.session_state:
        st.session_state.history = []
    

# Hàm load dữ liệu và setup
@st.cache_resource
def setup():

    openai.api_key = os.getenv("OPENAI_API_KEY")

    reranker = _get_reranker()

    
    vector_db = VectorDatabase(db_type= "mongodb")

    embedding = Embeddings(model_name= "text-embedding-3-large", type= "openai") 

    routes = [
        Route(name="info_CLB", samples=info_CLBSample),
        Route(name="chitchat", samples=chitchatSample)
    ]
    router = SemanticRouter(embedding, routes)
    
    return embedding, vector_db, router,reranker
    

#Hàm xử lí truy vấn người dùng
def handle_query(query, embedding, vector_db, reranker, router):

    openai.api_key = os.getenv("OPENAI_API_KEY")


    #Sử dụng semantic_router

    
    route_result = router.guide(query)
    best_route = route_result[1] # lấy tên router
    if best_route == "uncertain":
        best_route = "chitchat"
        messages_for_answer = [
            {"role": "system", "content": st.session_state.messages[0]["content"]},
            {"role": "user", "content": query}
        ]
    st.chat_message("assistant").markdown(f"**[Định tuyến]:** `{best_route}`")
    if best_route == "info_CLB":
        reflection = Reflection(openai)

        rewritten_query = reflection._rewrite(st.session_state.messages, query)
    
        #embedding câu hỏi
        # Tạo embedding cho câu hỏi của người dùng
        user_embedding = embedding.encode(rewritten_query)

            # Tìm kiếm thông tin liên quan trong cơ sở dữ liệu
        results = vector_db.query("information",user_embedding,limit= 60)
           
            # Rerank
        print("Kết quả trước rerank:")

        cnt = 0
        for result in results:
            print (f"Văn bản số: {cnt+1}")
            print(f"Title: {result['title']}")
            print (f"Information: {result['information']}")
            print("-" *50)
            cnt+=1
            if cnt == 5:
                break
        
        passages = [result["information"] for result in results]
        scores, reranker_passages = reranker(rewritten_query, passages)

            # (5) Map ngược passage -> result gốc để không mất title/metadata
            #    Dùng dict {passage: [list các index]} để xử lý trường hợp passage trùng nhau
        passage2idxs = defaultdict(list)
        for idx, p in enumerate(passages):
            passage2idxs[p].append(idx)

        reranked_results = []
        for s, p in zip(scores,  reranker_passages):
            if passage2idxs[p]:
                idx = passage2idxs[p].pop(0)   # lấy index đầu tiên khớp passage này
                r = results[idx].copy()        # copy để không sửa object gốc (tuỳ bạn)
                r["_rerank_score"] = float(s)
                reranked_results.append(r)
            
        reranked_results = reranked_results[:15]
        
            # In kết quả sau reranking
        dem = 0
        print("\n📊 Kết quả sau khi rerank:")
        for i, r in enumerate (reranked_results):
            print(f"Văn bản{i+1} | Score: {r.get('_rerank_score', 0):.4f}")
            print(r.get('information', ''))
            print("-" *50)
            dem+= 1
            if dem == 5:
                break
           

            #Ghép các đoạn tìm được thành một khối 'context' văn bản phẳng
        context = "\n".join(result["information"] for result in reranked_results)
        
        base_prompt = st.session_state.messages[0]["content"]
        
        new_context = base_prompt + f"\nCâu hỏi: {rewritten_query}\n" + f"\nThông tin liên quan:\n{context}"
        #st.session_state.messages[0]["content"] = new_context    
        
        messages_for_answer = [
            {"role": "system", "content": base_prompt},
            {"role": "user",
            "content": new_context}
        ]
         
    #else:
    #    st.session_state.messages.append({"role":"user", "content": query})
    
    # Gọi openAI dạng stream
    response_stream = embedding.client.chat.completions.create(
        model = "gpt-4o-mini",
        messages=messages_for_answer,
        stream=True
    )
    with st.chat_message("assistant"):
        assitant_reply = ""
        placeholder = st.empty()

        for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                assitant_reply += token
                placeholder.markdown(assitant_reply +"▌")
        placeholder.markdown(assitant_reply)
    st.session_state.messages.append({"role": "assistant", "content": assitant_reply})
    
    return None

       
    
#--- Giao diện Streamlit----

st.title("Chat Bot")

init_session()
embedding, vector_db,router,reranker = setup()


# Hiển thị hội thoại cũ
for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
       st.markdown(message["content"])

user_input = st.chat_input("Anh/chị muốn hỏi gì?")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    handle_query(user_input, embedding, vector_db,reranker, router)
    