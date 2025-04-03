"""
RAG Chatbot Tương Tác Liên Tục - Cho phép Q&A liên tục với dữ liệu đã tải
"""
from dotenv import load_dotenv
import os
import shutil
import time

# Import từ langchain_community thay vì langchain
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma  # Sửa import từ langchain_community
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Nạp biến môi trường từ .env
load_dotenv()
API = os.getenv('API_KEY')
os.environ["GOOGLE_API_KEY"] = API

class InteractiveRAGChatbot:
    def __init__(self, 
                 persist_directory: str = "./chroma_db",
                 model_name: str = "gemini-1.5-pro-latest", 
                 temperature: float = 0.2):
        """
        Khởi tạo RAG chatbot tương tác liên tục
        
        Args:
            persist_directory: Thư mục để lưu trữ vector database
            model_name: Tên model Gemini để sử dụng
            temperature: Giá trị temperature cho LLM (0.0-1.0)
        """
        # Lưu trữ thư mục persistance
        self.persist_directory = persist_directory
        
        # Khởi tạo embedding model
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Khởi tạo LLM
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            convert_system_message_to_human=True
        )
        
        # Khởi tạo text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Khởi tạo memory để lưu trữ lịch sử hội thoại
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Tải vector store nếu đã tồn tại, nếu không thì tạo mới
        self._initialize_vectorstore()
        
        # QA chain sẽ được khởi tạo khi cần
        self.qa_chain = None
    
    def _initialize_vectorstore(self):
        """Kiểm tra và tải vector store nếu đã tồn tại"""
        if os.path.exists(self.persist_directory) and os.path.isdir(self.persist_directory):
            try:
                print(f"Tìm thấy vector database tại {self.persist_directory}, đang tải...")
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                print(f"Đã tải vector database thành công")
                # Khởi tạo QA chain ngay khi tải vector store
                self._setup_qa_chain()
            except Exception as e:
                print(f"Lỗi khi tải vector database: {e}")
                print("Sẽ tạo mới vector database...")
                # Xóa thư mục có vấn đề
                shutil.rmtree(self.persist_directory)
                self.vectorstore = None
        else:
            print("Chưa có vector database, sẽ được tạo khi tải tài liệu đầu tiên")
            self.vectorstore = None
    
    def load_documents(self, file_paths, force_reload=False):
        """
        Tải một hoặc nhiều tài liệu vào vector store
        
        Args:
            file_paths: Đường dẫn đến file hoặc danh sách đường dẫn
            force_reload: Nếu True, xóa database cũ và tạo mới
            
        Returns:
            Số lượng chunks đã tải
        """
        # Xử lý tham số đường dẫn đơn lẻ
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        
        # Xóa database cũ nếu yêu cầu
        if force_reload and os.path.exists(self.persist_directory):
            print(f"Xóa vector database cũ tại {self.persist_directory}")
            shutil.rmtree(self.persist_directory)
            self.vectorstore = None
        
        # Tải và xử lý tất cả tài liệu
        all_chunks = []
        for file_path in file_paths:
            try:
                # Kiểm tra file tồn tại
                if not os.path.exists(file_path):
                    print(f"File không tồn tại: {file_path}")
                    continue
                
                print(f"Đang xử lý file: {file_path}")
                loader = TextLoader(file_path)
                documents = loader.load()
                
                # Chia nhỏ tài liệu
                chunks = self.text_splitter.split_documents(documents)
                print(f"  - Đã chia thành {len(chunks)} chunks")
                all_chunks.extend(chunks)
                
            except Exception as e:
                print(f"Lỗi khi xử lý file {file_path}: {str(e)}")
        
        # Nếu không có chunks nào được tạo
        if not all_chunks:
            print("Không có tài liệu nào được xử lý thành công")
            return 0
        
        # Tạo hoặc cập nhật vector store
        if self.vectorstore is None:
            print(f"Tạo mới vector database tại {self.persist_directory}")
            self.vectorstore = Chroma.from_documents(
                documents=all_chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            # Lưu xuống đĩa
            self.vectorstore.persist()
            print(f"Đã lưu vector database với {len(all_chunks)} chunks")
        else:
            print(f"Cập nhật vector database hiện có")
            self.vectorstore.add_documents(all_chunks)
            # Lưu xuống đĩa
            self.vectorstore.persist()
            print(f"Đã cập nhật vector database, thêm {len(all_chunks)} chunks")
        
        # Khởi tạo QA chain
        self._setup_qa_chain()
        
        return len(all_chunks)
    
    def load_directory(self, directory_path, extensions=['.txt', '.md', '.markdown']):
        """
        Tải tất cả các file với phần mở rộng được chỉ định từ một thư mục
        
        Args:
            directory_path: Đường dẫn đến thư mục
            extensions: Danh sách các phần mở rộng file cần tải
            
        Returns:
            Số lượng file đã tải
        """
        if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
            print(f"Thư mục không tồn tại: {directory_path}")
            return 0
        
        # Tìm tất cả các file phù hợp
        file_paths = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_paths.append(os.path.join(root, file))
        
        if not file_paths:
            print(f"Không tìm thấy file phù hợp trong thư mục {directory_path}")
            return 0
        
        print(f"Tìm thấy {len(file_paths)} file để xử lý")
        # Tải tất cả file tìm được
        loaded_chunks = self.load_documents(file_paths)
        
        return loaded_chunks
    
    def _setup_qa_chain(self):
        """Thiết lập QA chain với lịch sử hội thoại"""
        if self.vectorstore is None:
            print("Không thể thiết lập QA chain: chưa có vector store")
            return
        
        # Tạo retriever
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        # Tạo template cho prompt sử dụng cả lịch sử hội thoại
        template = """
        Bạn là trợ lý AI có kiến thức chuyên sâu. Nhiệm vụ của bạn là trả lời câu hỏi dựa trên các tài liệu.
        
        Lịch sử trò chuyện:
        {chat_history}
        
        Thông tin từ tài liệu:
        {context}
        
        Câu hỏi: {query}
        
        Trả lời bằng tiếng Việt, rõ ràng và ngắn gọn. Hãy dựa vào thông tin từ tài liệu.
        Nếu không tìm thấy thông tin trong dữ liệu, hãy nói rằng bạn không biết câu trả lời dựa trên dữ liệu hiện có.
        
        Câu trả lời:
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["chat_history", "context", "query"]
        )
        
        # Tạo QA chain
        try:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": prompt,
                    "memory": self.memory
                }
            )
            print("Đã khởi tạo QA chain thành công")
        except Exception as e:
            print(f"Lỗi khi khởi tạo QA chain: {e}")
            self.qa_chain = None
    
    def query(self, question, return_sources=False):
        """
        Truy vấn chatbot
        
        Args:
            question: Câu hỏi cần trả lời
            return_sources: Nếu True, trả về cả nguồn tài liệu
            
        Returns:
            Câu trả lời hoặc dict chứa câu trả lời và nguồn tài liệu
        """
        if self.vectorstore is None:
            return "Vui lòng tải tài liệu trước khi truy vấn."
        
        if self.qa_chain is None:
            self._setup_qa_chain()
            if self.qa_chain is None:
                return "Không thể khởi tạo QA chain."
        
        try:
            # Thực hiện truy vấn
            result = self.qa_chain({"query": question})
            
            # Trích xuất câu trả lời
            if "result" in result:
                answer = result["result"]
            else:
                # Tìm key phù hợp trong kết quả
                for key, value in result.items():
                    if isinstance(value, str) and key != "query":
                        answer = value
                        break
                else:
                    answer = "Không thể trích xuất câu trả lời từ kết quả."
            
            # Trả về kèm nguồn nếu yêu cầu
            if return_sources and "source_documents" in result:
                sources = []
                for doc in result["source_documents"]:
                    source_info = {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    sources.append(source_info)
                
                return {"answer": answer, "sources": sources}
            
            return answer
            
        except Exception as e:
            print(f"Lỗi khi thực hiện truy vấn: {str(e)}")
            return self.manual_query(question)
    
    def manual_query(self, question):
        """Phương pháp truy vấn thủ công"""
        if self.vectorstore is None:
            return "Vui lòng tải tài liệu trước khi truy vấn."
        
        try:
            # Tạo retriever
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 10}
            )
            
            # Lấy tài liệu liên quan
            docs = retriever.get_relevant_documents(question)
            
            # Tạo context từ các tài liệu
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Lấy lịch sử trò chuyện
            chat_history = self.memory.buffer_as_str if hasattr(self.memory, 'buffer_as_str') else ""
            
            # Tạo prompt
            prompt_text = f"""
            Bạn là trợ lý AI có kiến thức chuyên sâu. Nhiệm vụ của bạn là trả lời câu hỏi dựa trên các tài liệu.
            
            Lịch sử trò chuyện:
            {chat_history}
            
            Thông tin từ tài liệu:
            {context}
            
            Câu hỏi: {question}
            
            Trả lời bằng tiếng Việt, rõ ràng và ngắn gọn. Hãy dựa vào thông tin từ tài liệu.
            Nếu không tìm thấy thông tin trong dữ liệu, hãy nói rằng bạn không biết câu trả lời dựa trên dữ liệu hiện có.
            
            Câu trả lời:
            """
            
            # Gửi prompt đến LLM
            response = self.llm.invoke(prompt_text)
            answer = response.content
            
            # Cập nhật lịch sử
            self.memory.save_context({"input": question}, {"output": answer})
            
            return answer
            
        except Exception as e:
            return f"Lỗi khi xử lý truy vấn thủ công: {str(e)}"
    
    def reset_conversation(self):
        """Xóa lịch sử hội thoại"""
        self.memory.clear()
        print("Đã xóa lịch sử hội thoại")
    
    def run_interactive(self):
        """Chạy chế độ tương tác liên tục"""
        if self.vectorstore is None:
            print("Bạn cần tải tài liệu trước khi bắt đầu tương tác.")
            file_path = input("Nhập đường dẫn đến file hoặc thư mục chứa tài liệu: ")
            
            if os.path.isdir(file_path):
                self.load_directory(file_path)
            else:
                self.load_documents(file_path)
            
            if self.vectorstore is None:
                print("Không thể tải tài liệu. Vui lòng kiểm tra đường dẫn và thử lại.")
                return
        
        print("\n===== CHẾ ĐỘ HỎI ĐÁP LIÊN TỤC =====")
        print("Gõ 'exit' để thoát, 'clear' để xóa lịch sử trò chuyện, 'sources' để xem nguồn của câu trả lời cuối")
        
        show_sources = False
        last_result = None
        
        while True:
            question = input("\nHỏi: ")
            
            if question.lower() == 'exit':
                print("Cảm ơn bạn đã sử dụng chatbot!")
                break
            
            elif question.lower() == 'clear':
                self.reset_conversation()
                continue
            
            elif question.lower() == 'sources':
                if last_result and isinstance(last_result, dict) and 'sources' in last_result:
                    print("\n===== NGUỒN TÀI LIỆU =====")
                    for i, source in enumerate(last_result['sources']):
                        print(f"\nNguồn {i+1}:")
                        print(f"Nội dung: {source['content']}")
                        if 'metadata' in source:
                            print(f"Metadata: {source['metadata']}")
                else:
                    print("Không có thông tin về nguồn tài liệu cho câu trả lời gần nhất.")
                continue
            
            # Xử lý truy vấn
            start_time = time.time()
            
            if show_sources:
                result = self.query(question, return_sources=True)
                last_result = result
                answer = result['answer'] if isinstance(result, dict) else result
            else:
                answer = self.query(question)
                last_result = None
            
            end_time = time.time()
            
            # Hiển thị câu trả lời
            print(f"\nTrả lời ({round(end_time - start_time, 2)}s):")
            print(answer)


# Chạy chế độ tương tác
if __name__ == "__main__":
    # Khởi tạo chatbot
    chatbot = InteractiveRAGChatbot(persist_directory="./my_rag_db")
    
    # Tải tài liệu (bỏ comment dòng này nếu bạn muốn tải tài liệu ngay lập tức)
    # file_path = "results.md"  # Thay thế bằng đường dẫn của bạn
    # chatbot.load_documents(file_path)
    chatbot.load_directory("/home/pc/tienai/AI/RAG/markdown_files")
    # Chạy chế độ tương tác
    chatbot.run_interactive()