import asyncio
from crawl4ai import CrawlerWithBrowser
from crawl4ai.extractor import *

async def crawl_with_image_processing(url):
    async with CrawlerWithBrowser(
        headless=True,
        browser_args={"args": ["--no-sandbox"]}
    ) as crawler:
        # Cấu hình trích xuất ảnh tối ưu
        result = await crawler.arun(
            url=url,
            extract_rules={
                "images": {
                    "selector": "img",
                    "output": {
                        "src": "src",  # Ưu tiên lấy từ thuộc tính gốc
                        "data-src": "data-src",  # Fallback cho lazy loading
                        "alt": "alt||'image'"  # Mặc định 'image' nếu alt trống
                    },
                    "filter": lambda img: (
                        not img["src"].startswith("data:image") and 
                        "holder" not in img["src"]
                    )
                }
            },
            wait_for_selector="img",  # Đảm bảo ảnh đã tải xong
            timeout=15000  # Tăng timeout cho trang nặng
        )
        
        # Xử lý hậu kỳ (nếu cần)
        processed_markdown = result.markdown
        return processed_markdown

# Sử dụng
if __name__ == "__main__":
    url = "https://vndoc.com/toan-lop-4-bai-118-on-tap-chu-de-4-336846"
    markdown = asyncio.arun(crawl_with_image_processing(url))
    print(markdown)