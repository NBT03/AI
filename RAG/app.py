"""
Flask web interface cho RAG Chatbot
"""
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
import threading
import time

# Import RAG chatbot
from rag_chatbot import InteractiveRAGChatbot  # Giả sử bạn lưu code chatbot vào file rag_chatbot.py

# Nạp biến môi trường
load_dotenv()

app = Flask(__name__)

# Khởi tạo chatbot
chatbot = InteractiveRAGChatbot(persist_directory="./my_rag_db")

# Biến toàn cục để theo dõi trạng thái tải tài liệu
loading_status = {
    "is_loading": False,
    "total_files": 0,
    "processed_files": 0,
    "message": ""
}

@app.route('/')
def index():
    """Trang chủ"""
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def process_query():
    """Xử lý truy vấn"""
    data = request.json
    question = data.get('question', '')
    
    if not question:
        return jsonify({"error": "Câu hỏi không được để trống"}), 400
    
    # Kiểm tra xem có vector store chưa
    if chatbot.vectorstore is None:
        return jsonify({"error": "Chưa có dữ liệu nào được tải. Vui lòng tải dữ liệu trước."}), 400
    
    # Thực hiện truy vấn với sources
    try:
        result = chatbot.query(question, return_sources=True)
        
        if isinstance(result, dict) and 'answer' in result:
            response = {
                "answer": result['answer'],
                "has_sources": True,
                "sources": result['sources']
            }
        else:
            response = {
                "answer": result,
                "has_sources": False
            }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"Lỗi khi xử lý truy vấn: {str(e)}"}), 500

@app.route('/clear-history', methods=['POST'])
def clear_history():
    """Xóa lịch sử hội thoại"""
    chatbot.reset_conversation()
    return jsonify({"success": True, "message": "Đã xóa lịch sử hội thoại"})

def load_docs_thread(file_path, is_directory=False):
    """Hàm tải tài liệu trong thread riêng"""
    global loading_status
    
    loading_status["is_loading"] = True
    loading_status["message"] = "Đang bắt đầu tải dữ liệu..."
    
    try:
        if is_directory:
            loading_status["message"] = f"Đang quét thư mục {file_path}..."
            loaded_chunks = chatbot.load_directory(file_path)
            loading_status["message"] = f"Đã tải thành công {loaded_chunks} chunks từ thư mục."
        else:
            loading_status["message"] = f"Đang xử lý file {file_path}..."
            loaded_chunks = chatbot.load_documents([file_path])
            loading_status["message"] = f"Đã tải thành công {loaded_chunks} chunks từ file."
    except Exception as e:
        loading_status["message"] = f"Lỗi khi tải dữ liệu: {str(e)}"
    
    loading_status["is_loading"] = False

@app.route('/load-file', methods=['POST'])
def load_file():
    """API để tải file"""
    global loading_status
    
    if loading_status["is_loading"]:
        return jsonify({"error": "Đang trong quá trình tải dữ liệu. Vui lòng đợi."}), 400
    
    data = request.json
    file_path = data.get('file_path', '')
    is_directory = data.get('is_directory', False)
    
    if not file_path:
        return jsonify({"error": "Đường dẫn không được để trống"}), 400
    
    if not os.path.exists(file_path):
        return jsonify({"error": "Đường dẫn không tồn tại"}), 400
    
    # Tải file/thư mục trong thread riêng để không block server
    thread = threading.Thread(target=load_docs_thread, args=(file_path, is_directory))
    thread.daemon = True
    thread.start()
    
    return jsonify({"success": True, "message": "Đang bắt đầu tải dữ liệu..."})

@app.route('/loading-status', methods=['GET'])
def get_loading_status():
    """API để kiểm tra trạng thái tải dữ liệu"""
    return jsonify(loading_status)

@app.route('/reset-database', methods=['POST'])
def reset_database():
    """Xóa toàn bộ database"""
    global loading_status
    
    if loading_status["is_loading"]:
        return jsonify({"error": "Đang trong quá trình tải dữ liệu. Vui lòng đợi."}), 400
    
    try:
        chatbot.reset_database()
        return jsonify({"success": True, "message": "Đã xóa toàn bộ database"})
    except Exception as e:
        return jsonify({"error": f"Lỗi khi xóa database: {str(e)}"}), 500

if __name__ == '__main__':
    # Chạy ứng dụng trên tất cả các interface (0.0.0.0) để có thể truy cập từ bên ngoài
    # port=5000 là port mặc định của Flask
    app.run(host='0.0.0.0', port=5000, debug=True)