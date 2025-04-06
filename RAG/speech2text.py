import whisper

def transcribe_audio(audio_file_path, model_size="base"):
    """
    Chuyển đổi âm thanh thành văn bản sử dụng OpenAI Whisper.
    
    Tham số:
    audio_file_path (str): Đường dẫn đến file âm thanh cần chuyển đổi
    model_size (str): Kích thước mô hình (tiny, base, small, medium, large)
    
    Trả về:
    str: Văn bản được chuyển đổi
    """
    # Tải mô hình
    model = whisper.load_model(model_size)
    
    # Thực hiện chuyển đổi
    result = model.transcribe(audio_file_path)
    
    # Trả về kết quả
    return result["text"]

# Ví dụ sử dụng
if __name__ == "__main__":
    # Thay đổi đường dẫn đến file âm thanh của bạn
    audio_file = "output.mp3"
    
    # Chuyển đổi sử dụng mô hình base
    text = transcribe_audio(audio_file, model_size="tiny")
    
    print("Kết quả chuyển đổi:")
    print(text)