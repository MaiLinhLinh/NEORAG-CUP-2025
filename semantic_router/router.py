
import numpy as np

class SemanticRouter():
    def __init__ (self, embedding, routes):
        self.routes = routes
        self.embedding = embedding
        self.routesEmbedding = {} # từ điển lưu embedding của các samples

        for route in self.routes:
            self.routesEmbedding[route.name] = self.embedding.encode(route.samples)

    def get_routes(self):
        return self.routes
    
    def guide(self, query):
        queryEmbedding = self.embedding.encode(query)
        #Chuẩn hoá vector câu hỏi về độ dài 1 (L2-norm) để so sánh theo hướng cosine.
        queryEmbedding = queryEmbedding / np.linalg.norm(queryEmbedding) 
        scores = []
        for route in self.routes:

            routesEmbedding = self.routesEmbedding[route.name] 
            routesEmbedding = np.atleast_2d(routesEmbedding)
            routesEmbedding = routesEmbedding/ (np.linalg.norm(routesEmbedding, axis= 1, keepdims=True))
            score = np.mean(np.dot(routesEmbedding, queryEmbedding.T).flatten())
            scores.append((score, route.name))
        
        scores.sort(reverse=True)
        best_score, best_name = scores[0]
        second = scores[1][0] if len(scores) > 1 else -1.0
        if best_score < 0.25 or (best_score - second) < 0.05:
            return (best_score, "uncertain")
        return scores[0]




        
