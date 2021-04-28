import scrapy


class BlogCrawler(scrapy.spiders.CrawlSpiders):
    self.count = 0
    name = "WWF-crawler"
    allowed_domains = ["https://www.worldwildlife.org/pages/world-wildlife-magazine-archive",
                       "https://www.worldwildlife.org/magazine"]
    start_urls = ["https://www.worldwildlife.org/pages/world-wildlife-magazine-archive"]

    rules = (
        Rule(LinkExtractor(), callback="parse_item", follow=True)
    )

    def parse_item(self, response):
        filename = response.url.split("/")[-2] + ".html"
        with open(filename, "wb") as file:
            file.write(response.body)

