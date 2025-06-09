# Interface for Finnhub API (Financial News API as mentioned in analyze_tool.md) 

# Interface for accessing Finnhub API data.
# analyze_tool.md lists Finnhub for high-quality financial data, including real-time stock dynamics.
# Finnhub API provides endpoints for company news, general news, and news sentiment.

import finnhub # type: ignore
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv # Added for .env loading

load_dotenv() # Load environment variables from .env file

# API key is now primarily loaded from .env, with a fallback to the placeholder.
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY") or "YOUR_API_KEY_HERE"

# Global Finnhub client instance
finnhub_client_instance = None

def get_finnhub_client():
    global finnhub_client_instance
    if FINNHUB_API_KEY == "YOUR_API_KEY_HERE" or FINNHUB_API_KEY is None:
        print("错误：FINNHUB_API_KEY 未配置。请在 .env 文件中或直接在脚本中设置。")
        print("请访问 https://finnhub.io/dashboard 获取API密钥。")
        return None
    if finnhub_client_instance is None:
        try:
            finnhub_client_instance = finnhub.Client(api_key=FINNHUB_API_KEY)
        except Exception as e:
            print(f"初始化Finnhub客户端时发生错误: {e}")
            finnhub_client_instance = None
            return None
    return finnhub_client_instance

def get_company_news(symbol, from_date_str, to_date_str):
    """
    Fetches company-specific news from Finnhub.

    Args:
        symbol (str): Company symbol (e.g., "AAPL").
        from_date_str (str): Start date for news in "YYYY-MM-DD" format.
        to_date_str (str): End date for news in "YYYY-MM-DD" format.

    Returns:
        list: A list of news articles, or None if an error occurs.
    """
    client = get_finnhub_client()
    if not client:
        return None
    try:
        news = client.company_news(symbol, _from=from_date_str, to=to_date_str)
        if isinstance(news, dict) and news.get('error'):
            print(f"Finnhub API (company_news) 返回错误: {news['error']}")
            return None
        return news
    except finnhub.FinnhubAPIException as e:
        print(f"Finnhub API (company_news) 调用时发生错误: {e}")
        return None
    except Exception as e:
        print(f"处理Finnhub company_news数据时发生未知错误: {e}")
        return None

def get_general_news(category, min_id=0):
    """
    Fetches general news by category from Finnhub.

    Args:
        category (str): News category (e.g., "general", "forex", "crypto", "merger").
        min_id (int, optional): Use this field to look for news articles before this ID. Defaults to 0.

    Returns:
        list: A list of news articles, or None if an error occurs.
    """
    client = get_finnhub_client()
    if not client:
        return None
    try:
        news = client.general_news(category, min_id=min_id)
        if isinstance(news, dict) and news.get('error'):
            print(f"Finnhub API (general_news) 返回错误: {news['error']}")
            return None
        return news
    except finnhub.FinnhubAPIException as e:
        print(f"Finnhub API (general_news) 调用时发生错误: {e}")
        return None
    except Exception as e:
        print(f"处理Finnhub general_news数据时发生未知错误: {e}")
        return None

def get_news_sentiment(symbol):
    """
    Fetches news sentiment for a specific company from Finnhub.

    Args:
        symbol (str): Company symbol (e.g., "AAPL").

    Returns:
        dict: News sentiment data, or None if an error occurs.
    """
    client = get_finnhub_client()
    if not client:
        return None
    try:
        sentiment = client.news_sentiment(symbol)
        if isinstance(sentiment, dict) and sentiment.get('error'): 
            print(f"Finnhub API (news_sentiment) 返回错误: {sentiment['error']}")
            return None
        return sentiment
    except finnhub.FinnhubAPIException as e:
        print(f"Finnhub API (news_sentiment) 调用时发生错误: {e}")
        return None
    except Exception as e:
        print(f"处理Finnhub news_sentiment数据时发生未知错误: {e}")
        return None

# --- 基础测试 ---
if __name__ == "__main__":
    print("--- 测试 Finnhub API ---")

    client_init = get_finnhub_client()
    if not client_init:
        print("由于API密钥未配置或客户端初始化失败，跳过Finnhub API的实时测试。")
    else:
        today = datetime.now()
        one_week_ago = today - timedelta(days=7)
        from_date_test = one_week_ago.strftime("%Y-%m-%d")
        to_date_test = today.strftime("%Y-%m-%d")

        print(f"\n测试日期范围: {from_date_test} 到 {to_date_test}")

        print("\n测试1: 获取 AAPL (Apple Inc.) 的公司新闻")
        aapl_news = get_company_news("AAPL", from_date_test, to_date_test)
        if isinstance(aapl_news, list) and aapl_news:
            print(f"获取到 {len(aapl_news)} 条关于 AAPL 的新闻 (最多显示3条): ")
            for i, article in enumerate(aapl_news[:3]):
                print(f"  标题: {article.get('headline')}")
                print(f"  来源: {article.get('source')}")
                print(f"  URL: {article.get('url')}")
                print("----")
        elif aapl_news is not None: 
            if not isinstance(aapl_news, list):
                 print("获取AAPL公司新闻数据失败或返回非列表格式。")
                 print("API 响应:", json.dumps(aapl_news, indent=2))
            else:
                 print("未找到 AAPL 的公司新闻。") 
        else:
            print("未能获取AAPL的公司新闻 (函数返回None)。")

        print("\n测试2: 获取通用类别 ('general') 的新闻")
        general_news_list = get_general_news("general")
        if isinstance(general_news_list, list) and general_news_list:
            print(f"获取到 {len(general_news_list)} 条通用新闻 (最多显示3条): ")
            for i, article in enumerate(general_news_list[:3]):
                print(f"  标题: {article.get('headline')}")
                print(f"  来源: {article.get('source')}")
                print("----")
        elif general_news_list is not None:
            if not isinstance(general_news_list, list):
                print("获取通用新闻数据失败或返回非列表格式。")
                print("API 响应:", json.dumps(general_news_list, indent=2))
            else:
                print("未找到通用新闻。")
        else:
            print("未能获取通用新闻 (函数返回None)。")

        print("\n测试3: 获取 TSLA (Tesla Inc.) 的新闻情感")
        tsla_sentiment = get_news_sentiment("TSLA")
        if isinstance(tsla_sentiment, dict) and not tsla_sentiment.get('error'):
            print("获取到 TSLA 的新闻情感数据:")
            print(json.dumps(tsla_sentiment, indent=2))
        elif tsla_sentiment is not None:
            print("获取TSLA新闻情感数据失败或返回错误。")
            print("API 响应:", json.dumps(tsla_sentiment, indent=2))
        else:
            print("未能获取TSLA的新闻情感数据 (函数返回None)。")

    print("\n所有Finnhub测试结束。")
    print("注意: Finnhub的免费API密钥有请求频率限制。")
    print("如果测试失败或返回错误，请检查您的API密钥状态和使用限制。") 