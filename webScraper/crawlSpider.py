# DeepText Web Scraper
# Designed for the wwf blog webpage so features may not work on other sites
# Usage: scrapy runspider crawlSpider.py "allowed domain" "starting url"

# General Imports for file saving as html to be parsed
import sys
import os.path
from pathlib import Path


# Scrapy Imports
from abc import ABC

import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule


class BlogCrawler(CrawlSpider, ABC):
    name = "WWF-crawler"

    allowed_domains = ['www.worldwildlife.org']
    # allowed_domains = [sys.argv[0]]  # User inputs allowed domain in string form

    start_urls = ['https://www.worldwildlife.org/magazine/issues/summer-2021/articles/restoring-brazil-s-atlantic-forest',
            'https://www.worldwildlife.org/magazine/issues/spring-2021/articles/the-bezos-earth-fund-and-wwf-invest-in-solutions-for-the-climate-crisis'
            ]

    rules = (
        Rule(LinkExtractor(), callback="parse_item", follow=True),
    )

    # parses response from crawl spider and outputs plain html
    def parse_item(self, response):
        filename = response.url.split("/")[-2] + ".html"  # This needs to be changed as it keeps overwriting files
        # Response is plain html therefore output file is html that needs to be passed to the parser for text extraction
        completename = os.path.join('output/', filename)  # All files are placed in output folder, gitignore

        filetocheck = Path('output/' + filename)
        c = 0
        if filetocheck.is_file():
            filename = str(c) + filename
            completename = os.path.join('output/', filename)
            c = c + 1
        else:
            completename = os.path.join('output/', filename)

        with open(completename, "wb") as file:
            file.write(response.body)
