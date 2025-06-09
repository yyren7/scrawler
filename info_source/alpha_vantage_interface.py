# Interface for accessing Alpha Vantage API data.
# analyze_tool.md lists Alpha Vantage for financial data, including news and sentiments.

import requests
import json
import os # Added for os.environ.get
from dotenv import load_dotenv # Added for .env loading

load_dotenv() # Load environment variables from .env file

# API key is now primarily loaded from .env, with a fallback to the placeholder.
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY") or "YOUR_API_KEY_HERE" 
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

def get_alpha_vantage_news_sentiment(api_key=None, tickers=None, topics=None, time_from=None, time_to=None, sort="LATEST", limit=50):
    """
    Fetches market news and sentiment data from Alpha Vantage.

    Args:
        api_key (str, optional): Your Alpha Vantage API key. If None, uses the globally defined ALPHA_VANTAGE_API_KEY.
        tickers (str, optional): A comma-separated string of stock/crypto/forex symbols 
                                 (e.g., "AAPL,MSFT", "CRYPTO:BTC", "FOREX:USD"). Defaults to None (all tickers).
        topics (str, optional): A comma-separated string of news topics. 
                                (e.g., "technology,ipo", "earnings"). See API docs for full list. Defaults to None (all topics).
        time_from (str, optional): Start time for news articles in YYYYMMDDTHHMM format (e.g., "20220410T0130"). Defaults to None.
        time_to (str, optional): End time for news articles in YYYYMMDDTHHMM format. Defaults to None.
        sort (str, optional): Sort order. "LATEST" (default), "EARLIEST", or "RELEVANCE".
        limit (int, optional): Number of results to return (default 50, max 1000 for premium, free tier may have lower limits e.g. 50).

    Returns:
        dict: JSON response from the API, or None if an error occurs or API key is missing.
              The news feed is typically in response_dict[\'feed\'].
    """
    current_api_key = api_key or ALPHA_VANTAGE_API_KEY
    if current_api_key == "YOUR_API_KEY_HERE" or current_api_key is None:
        print("错误：ALPHA_VANTAGE_API_KEY 未配置。请在 .env 文件中或直接在脚本中设置。")
        print("请访问 https://www.alphavantage.co/support/#api-key 获取API密钥。")
        return None

    params = {
        "function": "NEWS_SENTIMENT",
        "apikey": current_api_key,
        "sort": sort,
        "limit": str(limit) 
    }

    if tickers:
        params["tickers"] = tickers
    if topics:
        params["topics"] = topics
    if time_from:
        params["time_from"] = time_from
    if time_to:
        params["time_to"] = time_to
    
    response = None
    try:
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params)
        response.raise_for_status() 
        
        parsed_response = response.json()
        
        if "Error Message" in parsed_response:
            print(f"Alpha Vantage API 返回错误: {parsed_response['Error Message']}")
            return None
        # The "Information" key often indicates a rate limit message for free tier.
        # The API might still return some data (e.g. an empty feed) or just the info message.
        if "Information" in parsed_response:
            print(f"Alpha Vantage API 返回信息: {parsed_response['Information']}")
            # If it's just an info message and no feed, it might be a hard limit hit.
            if 'feed' not in parsed_response and not parsed_response.get("items", ""): # items is another way AV returns count
                 return parsed_response # Return the info message itself for inspection

        # Check if 'feed' exists and is a list, or if it's an empty result but valid response structure
        if 'feed' not in parsed_response and parsed_response.get("items") == "0":
             print(f"Alpha Vantage: 日期 {time_from}-{time_to}、代码 {tickers}、主题 {topics} 无匹配新闻。")
             return parsed_response # Return the response, it's a valid no-data scenario
        elif 'feed' not in parsed_response:
            print("Alpha Vantage API响应格式不符合预期，缺少 'feed' 键。")
            print(f"请求参数: {params}")
            print(f"完整响应: {json.dumps(parsed_response, indent=2)}")
            return None
            
        return parsed_response

    except requests.exceptions.RequestException as e:
        print(f"调用Alpha Vantage API时发生错误: {e}")
        if response is not None:
            print(f"响应状态码: {response.status_code}")
            print(f"响应内容: {response.text}")
        return None
    except json.JSONDecodeError:
        print("解析Alpha Vantage API响应时发生JSON解码错误。")
        if response is not None:
            print(f"响应文本: {response.text}")
        return None
    except Exception as e:
        print(f"处理Alpha Vantage数据时发生未知错误: {e}")
        return None

# --- 基础测试 ---
if __name__ == "__main__":
    print("--- 测试 Alpha Vantage API ---")

    if ALPHA_VANTAGE_API_KEY == "YOUR_API_KEY_HERE" or ALPHA_VANTAGE_API_KEY is None:
        print("请先在 .env 文件或脚本中配置 ALPHA_VANTAGE_API_KEY 再运行测试。")
        print("跳过Alpha Vantage API的实时测试。")
    else:
        print("\n测试1: 获取关于 AAPL (Apple Inc.) 的最新新闻和情感 (最多5条)")
        # Free tier might have limitations on historical data access or number of tickers/topics
        # API documentation says default limit 50, max 1000. Free tier often much lower in practice for news.
        aapl_news = get_alpha_vantage_news_sentiment(tickers="AAPL", limit=5) 

        if aapl_news and isinstance(aapl_news.get('feed'), list):
            print(f"获取到 {len(aapl_news['feed'])} 条关于 AAPL 的新闻:")
            for article in aapl_news['feed']:
                print(f"  标题: {article.get('title')}")
                print(f"  来源: {article.get('source')}, ({article.get('source_domain')})")
                print(f"  发布时间: {article.get('time_published')}")
                print(f"  情感得分: {article.get('overall_sentiment_score')}, 标签: {article.get('overall_sentiment_label')}")
                if article.get('ticker_sentiment'):
                    for ticker_sentiment in article.get('ticker_sentiment', []):
                        print(f"    -> Ticker: {ticker_sentiment.get('ticker')}, Relevance: {ticker_sentiment.get('relevance_score')}, Sentiment: {ticker_sentiment.get('ticker_sentiment_score')} ({ticker_sentiment.get('ticker_sentiment_label')})")
                print("----")
        elif aapl_news: # If not None, but feed isn't a list as expected
            print("获取AAPL新闻数据成功，但响应数据格式可能不正确或指示无数据/已达限制。")
            print("API 响应:", json.dumps(aapl_news, indent=2))
            if "Information" in aapl_news:
                print("提示: Alpha Vantage的免费套餐对于NEWS_SENTIMENT功能有严格的限制，例如每日请求次数或特定参数组合。")
        else:
            print("未能获取AAPL的新闻和情感数据。")

        print("\n测试2: 获取关于 '特斯拉' (TSLA) 且主题为 'earnings' 的新闻 (最多3条)")
        tsla_earnings_news = get_alpha_vantage_news_sentiment(tickers="TSLA", topics="earnings", limit=3)
        
        if tsla_earnings_news and isinstance(tsla_earnings_news.get('feed'), list):
            print(f"获取到 {len(tsla_earnings_news['feed'])} 条关于 TSLA 和 earnings 的新闻:")
            for article in tsla_earnings_news['feed']:
                print(f"  标题: {article.get('title')}")
                print(f"  URL: {article.get('url')}")
                print("----")
        elif tsla_earnings_news:
            print("获取TSLA earnings新闻数据成功，但响应数据格式可能不正确或指示无数据/已达限制。")
            print("API 响应:", json.dumps(tsla_earnings_news, indent=2))
            if "Information" in tsla_earnings_news:
                print("提示: Alpha Vantage的免费套餐对于NEWS_SENTIMENT功能有严格的限制。")
        else:
            print("未能获取TSLA的earnings新闻数据。")

        print("\n测试3: 尝试获取超出免费额度可能触发的信息提示 (此测试依赖于API KEY已达到当日限额)")
        # This test is more likely to show an "Information" message if run after other tests on free tier
        # We are using a generic topic that should have news, but asking for a higher limit
        # Note: The actual limit for free tier on NEWS_SENTIMENT can be very low (e.g. even 1 can sometimes hit daily limit if used before)
        many_news = get_alpha_vantage_news_sentiment(topics="technology", limit=10) # Attempt to get more
        if many_news and "Information" in many_news:
            print("成功触发 'Information' 消息 (可能表示已达免费限额或参数组合限制):")
            print(json.dumps(many_news, indent=2))
        elif many_news and isinstance(many_news.get('feed'), list) and len(many_news['feed']) > 0:
            print(f"获取到 {len(many_news['feed'])} 条 'technology' 新闻，未触发 'Information' 消息。API密钥可能仍有额度或此查询组合允许。")
        else:
            print("获取 'technology' 新闻测试未按预期触发 'Information' 消息，或获取失败。")
            if many_news: print("API 响应:", json.dumps(many_news, indent=2))


    print("\n所有Alpha Vantage测试结束。")
    print("注意: Alpha Vantage的免费API密钥有请求频率和总量的限制。")
    print("NEWS_SENTIMENT 端点在免费层级下限制可能尤其严格。")
    print("如果测试失败或返回 'Information' 消息，请检查您的API密钥状态和使用限制。") 