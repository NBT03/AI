import asyncio
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
    
async def main():
    browser_config = BrowserConfig()  # Default browser configuration
    run_config = CrawlerRunConfig()   # Default crawl run configuration

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
        url="https://www.vietjack.com/soan-van-lop-6-kn/tom-tat-neu-cau-muon-co-mot-nguoi-ban.jsp",
        config=CrawlerRunConfig()
)

    print(result.markdown)
    with open("results.md", "w", encoding="utf-8") as file:
        file.write(result.markdown)

if __name__ == "__main__":
    asyncio.run(main())