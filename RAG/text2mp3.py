from gtts import gTTS

# Nhập nội dung văn bản cần chuyển thành giọng nói
text = "Xin chào, đây là ví dụ sử dụng gTTS để tạo file âm thanh từ văn bản."

# Tạo đối tượng gTTS
tts = gTTS(text=text, lang='vi')  # lang='vi' là tiếng Việt

# Lưu file mp3
tts.save("output.mp3")

print("Đã tạo xong file output.mp3")