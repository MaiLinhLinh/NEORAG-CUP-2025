from typing import List, Dict
class Reflection:
    def __init__ (self, llm_client):
        # llm_client là openai client đã khởi tạo
        self.llm_client = llm_client

    def _rewrite(self, messages: List[Dict], current_query: str) -> str:
        """
        Viết lại current_query thành câu hỏi độc lập từ context.

        :param messages: Lịch sử chat (dạng OpenAI chat messages)
        :param current_query: Câu hỏi hiện tại từ người dùng
        :return: Câu hỏi đã viết lại
        """

        # Lấy 10 messages gần nhất không phải roke = system
        chat_history = [msg for msg in messages if msg['role'] in ('user','assistant')][-5:]

        history_text = ""
        for msg in chat_history:
            role = "Khách" if msg['role'] == "user" else breakpoint
            history_text += f"{role}: {msg['content']}\n"
        history_text += f"Khách: {current_query}"

        prompt = [
            {
                "role": "system",
                "content": """Bạn là trợ lí AI thông minh của CLB Lập Trình PTIT. Bạn được cung cấp lịch sử hội thoại và câu hỏi mới nhất của người dùng (có thể tham chiếu đến ngữ cảnh trong lịch sử). Hãy soạn một câu hỏi độc lập có thể được hiểu mà không cần lịch sử trò chuyện.

                **Yêu cầu tạo câu hỏi**: 
                1. Không trả lời câu hỏi.
                2. Bám sát câu hỏi, chỉ bổ sung ngữ cảnh nếu cần thiết.
                3. 2) Giữ nguyên từ để hỏi ở đầu (ví dụ: "Ai", "Cái gì", "Khi nào", "Tại sao", "Làm sao"...). Được phép viết lại thông tin sau đó để dễ hiểu hơn nhưng không được tự bịa thông tin mới.
                4. KHÔNG gộp nội dung từ câu hỏi trước nếu khác chủ đề.
                5. KHÔNG TỰ BỊA CÂU HỎI.

                - Lưu ý: Nếu câu hỏi mới của người dùng KHÔNG LIÊN QUAN đến ngữ cảnh cung cấp thì giữ nguyên câu hỏi người dùng.
                """
             
            },
            {
                "role":"user",
                "content": history_text
            }
        ]

        # gọi LLM để rewrite câu hỏi
        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=prompt,
            temperature=0,
        )

        rewritten = response.choices[0].message.content.strip()
        print(f"Reflection: {rewritten}")
        return rewritten

