# Interface for accessing Polygon.io API data.
# analyze_tool.md lists Polygon.io for institutional-grade data, including financial news with sentiment.

import os
import json
from datetime import datetime, timedelta
from polygon import RESTClient # type: ignore
from polygon.exceptions import NoResultsError, BadResponse # type: ignore
from dotenv import load_dotenv # Added for .env loading

load_dotenv() # Load environment variables from .env file

# API key is now primarily loaded from .env, with a fallback to the placeholder.
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY") or "YOUR_POLYGON_API_KEY"

# Global Polygon.io client instance
polygon_client_instance = None

def get_polygon_client():
    global polygon_client_instance
    if POLYGON_API_KEY == "YOUR_POLYGON_API_KEY" or POLYGON_API_KEY is None:
        print("错误：POLYGON_API_KEY 未配置。请在 .env 文件中或直接在脚本中设置API密钥。")
        print("请访问 https://polygon.io/dashboard 获取API密钥。")
        return None
    if polygon_client_instance is None:
        try:
            polygon_client_instance = RESTClient(api_key=POLYGON_API_KEY)
        except Exception as e:
            print(f"初始化Polygon.io客户端时发生错误: {e}")
            polygon_client_instance = None 
            return None
    return polygon_client_instance

def get_ticker_news(symbol, limit=10, order="descending", sort="published_utc"):
    """
    Fetches news articles for a specific ticker symbol from Polygon.io.

    Args:
        symbol (str): The ticker symbol (e.g., "AAPL").
        limit (int, optional): Limit the number of results. Max 1000. Defaults to 10.
        order (str, optional): Order of results ("ascending" or "descending"). Defaults to "descending".
        sort (str, optional): Field to sort by (e.g., "published_utc"). Defaults to "published_utc".

    Returns:
        list: A list of news articles (each as a dict-like object), or None if an error occurs.
    """
    client = get_polygon_client()
    if not client:
        return None
    
    try:
        news_articles = client.get_ticker_news(ticker=symbol, limit=limit, order=order, sort=sort)
        results_list = [article for article in news_articles]
        
        if not results_list:
            print(f"Polygon.io: 未找到股票代码 {symbol} 的新闻。")
            return [] 
            
        return results_list
        
    except NoResultsError:
        print(f"Polygon.io: 未找到股票代码 {symbol} 的新闻 (NoResultsError)。")
        return [] 
    except BadResponse as e:
        print(f"Polygon.io API (get_ticker_news) 返回错误: {e}")
        if e.status == 401 or e.status == 403:
            print("提示: Polygon.io API密钥无效或权限不足。请检查您的API密钥和订阅计划。")
        elif e.status == 429:
            print("提示: 已达到Polygon.io的请求速率限制。")
        return None
    except Exception as e:
        print(f"处理Polygon.io get_ticker_news数据时发生未知错误: {e}")
        return None

# --- 基础测试 ---
if __name__ == "__main__":
    print("--- 测试 Polygon.io API ---")

    client_init_success = bool(get_polygon_client())

    if not client_init_success:
        print("由于API密钥未配置或客户端初始化失败，跳过Polygon.io API的实时测试。")
    else:
        print(f"使用的API密钥: {'已配置 (请确保其有效并具有所需权限)' if POLYGON_API_KEY != 'YOUR_POLYGON_API_KEY' else '未配置或为占位符'}")
        print("注意: Polygon.io的免费计划可能对数据访问和端点有限制。新闻数据通常需要付费订阅。")

        print("\n测试1: 获取 AAPL (Apple Inc.) 的最新3条新闻")
        aapl_news = get_ticker_news("AAPL", limit=3)

        if aapl_news is not None and isinstance(aapl_news, list):
            if aapl_news:
                print(f"获取到 {len(aapl_news)} 条关于 AAPL 的新闻:")
                for i, article_obj in enumerate(aapl_news):
                    print(f"  标题: {getattr(article_obj, 'title', 'N/A')}")
                    print(f"  发布者: {getattr(article_obj, 'publisher', {}).get('name', 'N/A')}")
                    article_url = getattr(article_obj, 'article_url', getattr(article_obj, 'amp_url', 'N/A'))
                    print(f"  URL: {article_url}")
                    published_utc = getattr(article_obj, 'published_utc', 'N/A')
                    print(f"  发布时间 (UTC): {published_utc}")
                    print("----")
            else:
                print("未找到 AAPL 的新闻，或API密钥无权访问此数据/已达限额。")
        else:
            print("未能获取AAPL的新闻。请检查API密钥和网络连接，或查看之前的错误消息。")

        print("\n测试2: 获取一个不存在的股票代码 (e.g., 'NONEXISTENTTICKERXYZ') 的新闻")
        non_existent_news = get_ticker_news("NONEXISTENTTICKERXYZ", limit=2)
        if non_existent_news is not None and isinstance(non_existent_news, list) and not non_existent_news:
            print("成功：对于不存在的股票代码，未获取到新闻 (返回空列表，符合预期)。")
        elif non_existent_news is None:
             print("对于不存在的股票代码，函数返回None (可能由于API错误或无结果错误被捕获)。")
        else:
            print("测试失败：对于不存在的股票代码，获取到了意外的响应:")
            try:
                serializable_list = []
                for item in non_existent_news:
                    if hasattr(item, '__dict__'):
                        serializable_list.append(item.__dict__)
                    else:
                        serializable_list.append(str(item)) 
                print(json.dumps(serializable_list, indent=2, default=str))
            except Exception as ser_e:
                print(f"无法序列化响应进行打印: {ser_e}")
                print(non_existent_news)

    print("\n所有Polygon.io测试结束。")
    print("提醒: Polygon.io的免费计划对数据和API端点的访问有限制。新闻数据通常需要付费订阅才能获取有意义的结果。") 