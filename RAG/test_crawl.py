import asyncio
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
    
async def main():
    browser_config = BrowserConfig()  # Default browser configuration
    run_config = CrawlerRunConfig()   # Default crawl run configuration

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
        url="https://vndoc.com/cam-nghi-cua-em-ve-mot-danh-lam-thang-canh-cua-que-huong-dat-nuoc-248073",
        config=CrawlerRunConfig()
)

    print(result.markdown)
    with open("results1.md", "w", encoding="utf-8") as file:
        file.write(result.markdown)

if __name__ == "__main__":
    asyncio.run(main())