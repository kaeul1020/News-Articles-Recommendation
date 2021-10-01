from korea_news_crawler.articlecrawler import ArticleCrawler

if __name__ == "__main__":
    Crawler = ArticleCrawler()  
    Crawler.set_category("정치", "IT과학", "economy", "사회", "생활문화")  
    Crawler.set_date_range(2017, 1, 2018, 4)  
    Crawler.start()