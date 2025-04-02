from dotenv import load_dotenv
from google import genai
import os
import google.generativeai as genai


# Tải các biến môi trường từ tệp .env
load_dotenv()

# Truy cập vào biến môi trường
API = os.getenv('API_KEY')

genai.configure(api_key=API)

# Lấy danh sách model khả dụng
# models = genai.list_models()

# # In danh sách model để kiểm tra
# for model in models:
model = genai.GenerativeModel("gemini-1.5-pro-latest")

def ask_gemini(question):
    response = model.generate_content(question)
    return response.text

# Test hỏi đáp
question = "cho tôi biết giới hạn call api của bạn là bao nhiêu?"
answer = ask_gemini(question)
print(answer)
