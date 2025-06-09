# Interface for accessing NewsAPI.org data.
# analyze_tool.md lists NewsAPI.org for fetching news articles.

import os
import json
from datetime import datetime, timedelta
from newsapi import NewsApiClient # type: ignore
from dotenv import load_dotenv # Added for .env loading

load_dotenv() # Load environment variables from .env file

# API key is now primarily loaded from .env, with a fallback to the placeholder.
NEWSAPI_ORG_API_KEY = os.environ.get("NEWSAPI_ORG_API_KEY") or "YOUR_API_KEY_HERE"

# Global NewsAPI.org client instance
newsapi_client_instance = None

def get_newsapi_org_client():
    global newsapi_client_instance
    if NEWSAPI_ORG_API_KEY == "YOUR_API_KEY_HERE" or NEWSAPI_ORG_API_KEY is None:
        print("错误：NEWSAPI_ORG_API_KEY 未配置。请在 .env 文件中或直接在脚本中设置API密钥。")
        print("请访问 https://newsapi.org/account 获取API密钥。")
        return None
    if newsapi_client_instance is None:
        try:
            newsapi_client_instance = NewsApiClient(api_key=NEWSAPI_ORG_API_KEY)
        except Exception as e:
            print(f"初始化NewsAPI.org客户端时发生错误: {e}")
            newsapi_client_instance = None
            return None
    return newsapi_client_instance

def get_everything(q=None, sources=None, domains=None, from_param=None, to_param=None, language='en', sort_by='publishedAt', page_size=20, page=1):
    """
    Search through millions of articles from over 150,000 large and small news sources and blogs.

    Args:
        q (str, optional): Keywords or phrases to search for in the article title and body.
        sources (str, optional): A comma-seperated string of identifiers for the news sources or blogs you want headlines from.
        domains (str, optional): A comma-seperated string of domains (eg bbc.co.uk, techcrunch.com) to restrict the search to.
        from_param (str, optional): A date and optional time for the oldest article allowed. ISO 8601 format (e.g., "2024-06-10" or "2024-06-10T10:00:00").
        to_param (str, optional): A date and optional time for the newest article allowed. ISO 8601 format.
        language (str, optional): The 2-letter ISO-639-1 code of the language you want to get headlines for. Defaults to 'en'.
        sort_by (str, optional): The order to sort the articles in. Possible options: "relevancy", "popularity", "publishedAt". Defaults to "publishedAt".
        page_size (int, optional): The number of results to return per page (request). 20 is the default, 100 is the maximum.
        page (int, optional): Use this to page through the results. Defaults to 1.

    Returns:
        dict: API response containing articles, or None if an error occurs.
    """
    client = get_newsapi_org_client()
    if not client:
        return None
    try:
        # Note: The NewsApiClient uses `from_param` for the `from` date due to `from` being a Python keyword.
        articles_response = client.get_everything(
            q=q, 
            sources=sources, 
            domains=domains, 
            from_param=from_param, 
            to=to_param, # `to` is acceptable as a parameter name in the client library
            language=language, 
            sort_by=sort_by, 
            page_size=page_size, 
            page=page
        )
        if articles_response.get("status") == "error":
            print(f"NewsAPI.org (get_everything) 返回错误: {articles_response.get('code')} - {articles_response.get('message')}")
            return None
        return articles_response
    except Exception as e:
        print(f"调用NewsAPI.org (get_everything) 时发生错误: {e}")
        # The client library might raise exceptions for certain HTTP errors or API errors.
        # Check the type of exception if more specific handling is needed.
        return None

def get_top_headlines(q=None, sources=None, category=None, language='en', country='us', page_size=20, page=1):
    """
    Provides live top and breaking headlines for a country, specific category in a country, single source, or multiple sources.

    Args:
        q (str, optional): Keywords or phrases to search for in the article title and body.
        sources (str, optional): A comma-seperated string of identifiers for the news sources or blogs you want headlines from.
        category (str, optional): The category you want to get headlines for. 
                                Possible options: business, entertainment, general, health, science, sports, technology.
        language (str, optional): The 2-letter ISO-639-1 code of the language. Defaults to 'en'.
        country (str, optional): The 2-letter ISO 3166-1 code of the country. Defaults to 'us'.
        page_size (int, optional): The number of results to return per page. Defaults to 20.
        page (int, optional): Use this to page through the results. Defaults to 1.

    Returns:
        dict: API response containing articles, or None if an error occurs.
    """
    client = get_newsapi_org_client()
    if not client:
        return None
    try:
        headlines_response = client.get_top_headlines(
            q=q, 
            sources=sources, 
            category=category, 
            language=language, 
            country=country, 
            page_size=page_size, 
            page=page
        )
        if headlines_response.get("status") == "error":
            print(f"NewsAPI.org (get_top_headlines) 返回错误: {headlines_response.get('code')} - {headlines_response.get('message')}")
            return None
        return headlines_response
    except Exception as e:
        print(f"调用NewsAPI.org (get_top_headlines) 时发生错误: {e}")
        return None

# --- 基础测试 ---
if __name__ == "__main__":
    print("--- 测试 NewsAPI.org API ---")

    client_init_success = bool(get_newsapi_org_client())

    if not client_init_success:
        print("由于API密钥未配置或客户端初始化失败，跳过NewsAPI.org API的实时测试。")
    else:
        print(f"使用的API密钥: {'已配置' if NEWSAPI_ORG_API_KEY != 'YOUR_API_KEY_HERE' else '未配置或为占位符'}")
        print("注意: NewsAPI.org 的免费套餐文章有24小时延迟，且每日有请求限制。")

        print("\n测试1: 获取关于 'Tesla' 的最新新闻 (最多3条)")
        tesla_news = get_everything(q="Tesla", page_size=3, sort_by="publishedAt")
        if tesla_news and tesla_news.get('status') == 'ok' and isinstance(tesla_news.get('articles'), list):
            print(f"获取到 {len(tesla_news['articles'])} 条关于 Tesla 的新闻:")
            for article in tesla_news['articles']:
                print(f"  标题: {article.get('title')}")
                print(f"  来源: {article.get('source', {}).get('name')}")
                print(f"  发布时间: {article.get('publishedAt')}")
                print(f"  URL: {article.get('url')}")
                print("----")
        else:
            print("未能获取Tesla的新闻。")
            if tesla_news: print("API 响应:", json.dumps(tesla_news, indent=2, ensure_ascii=False))

        print("\n测试2: 获取美国商业类的头条新闻 (最多3条)")
        us_business_headlines = get_top_headlines(country="us", category="business", page_size=3)
        if us_business_headlines and us_business_headlines.get('status') == 'ok' and isinstance(us_business_headlines.get('articles'), list):
            print(f"获取到 {len(us_business_headlines['articles'])} 条美国商业头条新闻:")
            for article in us_business_headlines['articles']:
                print(f"  标题: {article.get('title')}")
                print(f"  来源: {article.get('source', {}).get('name')}")
                print("----")
        else:
            print("未能获取美国商业头条新闻。")
            if us_business_headlines: print("API 响应:", json.dumps(us_business_headlines, indent=2, ensure_ascii=False))
        
        # Test with a past date (free tier has 24h delay, so recent news might not be available)
        # To test historical data (developer plan and above needed for more than 24h ago if not using q for all time)
        # For free tier, `from_param` and `to_param` will be most effective with `q` to search within the allowed window.
        print("\n测试3: 获取特定日期范围 'bitcoin' 的新闻")
        from_date_query = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d") 
        # to_date_query = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d") # up to yesterday due to 24h delay
        to_date_query = from_date_query # Search for a single day
        
        print(f"(注意: 免费版数据有延迟，查询日期为 {from_date_query})")
        bitcoin_news_dated = get_everything(q="bitcoin", from_param=from_date_query, to_param=to_date_query, page_size=2, sort_by="popularity")
        if bitcoin_news_dated and bitcoin_news_dated.get('status') == 'ok' and isinstance(bitcoin_news_dated.get('articles'), list):
            print(f"获取到 {len(bitcoin_news_dated['articles'])} 条关于 bitcoin 的新闻 ({from_date_query}):")
            for article in bitcoin_news_dated['articles']:
                print(f"  标题: {article.get('title')}")
                print(f"  来源: {article.get('source', {}).get('name')}")
                print("----")
        else:
            print(f"未能获取 {from_date_query} 的bitcoin新闻。")
            if bitcoin_news_dated: print("API 响应:", json.dumps(bitcoin_news_dated, indent=2, ensure_ascii=False))


    print("\n所有 NewsAPI.org 测试结束。")
    print("提醒: NewsAPI.org 的免费套餐有每日100次请求限制和24小时文章延迟。付费计划提供更多功能和实时数据。") 