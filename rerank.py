
from sentence_transformers import CrossEncoder
import numpy as np

# class Reranker():
#     def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
#         self.reranker = CrossEncoder(model_name, trust_remote_code=True)

#     def __call__(self, query: str, passages: list[str]) -> tuple[list[float], list[str]]:
#         # Combine query and passages into pairs
#         query_passage_pairs = [[query, passage] for passage in passages]

#         # Get scores from the reranker model
#         scores = self.reranker.predict(query_passage_pairs)

#         # Sort passages based on scores
#         ranked_passages = [passage for _, passage in sorted(zip(scores, passages), key=lambda x: x[0], reverse=True)]
#         ranked_scores = sorted(scores, reverse=True)
        
#         # Convert scores to standard Python floats
#         ranked_scores = [float(score) for score in ranked_scores]
#         # Return just the passages in ranked order
#         return ranked_scores, ranked_passages
    

# Model anh Nam

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,use_fast=True, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        trust_remote_code=True,
        
        )
        self.model.eval()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    '''
    def __call__(self, query: str, passages: list[str], batch_size: int = 8):
        # Tạo cặp [query, passage] cho mỗi passage
        query_passage_pairs = [[query, passage] for passage in passages]
        scores = []
        # Tính điểm từ reranker model
        with torch.no_grad():
            for i in range(0, len(query_passage_pairs), batch_size):
                batch = query_passage_pairs[i:i+batch_size]
                features  = self.tokenizer(batch, padding=True, truncation="longest_first", return_tensors="pt", max_length=256).to(self.device)
                model_predictions = self.model(**features, return_dict=True)

                logits = model_predictions.logits.view(-1, ).float()
                scores.extend(logits.cpu().numpy())


        # Sắp xếp passage theo điểm số giảm dần
        ranked_data = sorted(zip(scores, passages), key=lambda x: x[0], reverse=True)
        ranked_scores, ranked_passages = zip(*ranked_data)
        # Đảm bảo đầu ra là list chuẩn
        return list(ranked_scores), list(ranked_passages)

    '''
    
    def __call__(self, query: str, passages: list[str], batch_size: int = 8):
        # Tạo cặp [query, passage] cho mỗi passage
        query_passage_pairs = [[query, passage] for passage in passages]
        scores = []
        # Tính điểm từ reranker model
        with torch.no_grad():
            for i in range(0, len(query_passage_pairs), batch_size):
                batch = query_passage_pairs[i:i+batch_size]
                inputs = self.tokenizer(batch, padding=True, truncation=True,
                                        return_tensors='pt', max_length=512).to(self.device)
                logits = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
                scores.extend(logits.cpu().numpy())

        # Sắp xếp passage theo điểm số giảm dần
        ranked_data = sorted(zip(scores, passages), key=lambda x: x[0], reverse=True)
        ranked_scores, ranked_passages = zip(*ranked_data)
        # Đảm bảo đầu ra là list chuẩn
        return list(ranked_scores), list(ranked_passages)
    