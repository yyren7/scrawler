# Interface for accessing IEX Cloud API data.
# analyze_tool.md lists IEX Cloud for financial data, suitable for web apps and data scientists.
# IEX Cloud provides a news endpoint.

import pyex # type: ignore
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv # Added for .env loading

load_dotenv() # Load environment variables from .env file

# API key is now primarily loaded from .env, with a fallback to the placeholder.
# pyex library typically expects the environment variable to be IEX_TOKEN.
IEX_CLOUD_API_KEY = os.environ.get("IEX_TOKEN") or "YOUR_IEX_CLOUD_API_KEY_HERE"
IEX_CLOUD_API_VERSION = "stable" # or "v1", "beta", etc.

# Global IEX Cloud client instance
iex_client_instance = None

def get_iex_client():
    global iex_client_instance
    if IEX_CLOUD_API_KEY == "YOUR_IEX_CLOUD_API_KEY_HERE" or IEX_CLOUD_API_KEY is None:
        print("错误：IEX_TOKEN 未配置。请在 .env 文件中或直接在脚本中设置 IEX_CLOUD_API_KEY (环境变量IEX_TOKEN)。")
        print("请访问 https://iexcloud.io/console/ 获取API密钥。")
        return None
    if iex_client_instance is None:
        try:
            iex_client_instance = pyex.Client(api_token=IEX_CLOUD_API_KEY, version=IEX_CLOUD_API_VERSION)
        except Exception as e:
            print(f"初始化IEX Cloud客户端时发生错误: {e}")
            iex_client_instance = None 
            return None
    return iex_client_instance

def get_company_news_iex(symbol, count=5):
    """
    Fetches company-specific news from IEX Cloud.

    Args:
        symbol (str): Stock symbol (e.g., "AAPL").
        count (int, optional): Number of news articles to return (1-50). Defaults to 5.

    Returns:
        list: A list of news articles (each as a dict), or None if an error occurs.
    """
    client = get_iex_client()
    if not client:
        return None
    
    if not (1 <= count <= 50):
        print("错误: 新闻条数 (count) 必须在 1 到 50 之间。")
        return None

    try:
        news_list = client.news(symbol=symbol, count=count)
        
        if news_list is None: 
            print(f"IEX Cloud API (news) 未找到符号 {symbol} 的新闻或返回了None。")
            return None
        if isinstance(news_list, dict) and news_list.get('error'):
             print(f"IEX Cloud API (news) 返回错误: {news_list['error']}")
             return None
        return news_list
    except pyex.common.PyEXception as e: 
        print(f"IEX Cloud API (news) 调用时发生错误: {e}")
        if "response code 401" in str(e).lower() or "response code 403" in str(e).lower():
            print("提示: 收到的错误可能是由于API密钥无效或权限不足。请检查您的IEX Cloud API密钥和订阅计划。")
        elif "response code 402" in str(e).lower() or "response code 429" in str(e).lower():
             print("提示: 已达到IEX Cloud的消息/请求限制。免费套餐限制较高，付费套餐按消息计费。")
        return None
    except Exception as e:
        print(f"处理IEX Cloud company_news数据时发生未知错误: {e}")
        return None

# --- 基础测试 ---
if __name__ == "__main__":
    print("--- 测试 IEX Cloud API ---")

    client_init_success = bool(get_iex_client()) 

    if not client_init_success:
        print("由于API密钥未配置或客户端初始化失败，跳过IEX Cloud API的实时测试。")
    else:
        print(f"使用的API密钥类型: {'Sandbox/Test (Tpk_ or Tsk_开头的密钥通常返回混淆数据)' if IEX_CLOUD_API_KEY.startswith('T') else 'Production/Paid (pk_开头的密钥返回真实数据)'}")
        
        print("\n测试1: 获取 AAPL (Apple Inc.) 的最新5条公司新闻")
        aapl_news = get_company_news_iex("AAPL", count=5)

        if aapl_news and isinstance(aapl_news, list):
            print(f"获取到 {len(aapl_news)} 条关于 AAPL 的新闻:")
            for i, article in enumerate(aapl_news):
                print(f"  标题: {article.get('headline')}")
                print(f"  来源: {article.get('source')}")
                print(f"  URL: {article.get('url')}")
                ts = article.get('datetime')
                if ts:
                    try:
                        readable_time = datetime.fromtimestamp(ts / 1000.0).strftime('%Y-%m-%d %H:%M:%S')
                        print(f"  时间: {readable_time}")
                    except:
                        print(f"  时间戳: {ts} (无法解析)")
                print("----")
        elif aapl_news is not None: 
            print("获取AAPL公司新闻数据可能存在问题。")
            print("API 响应:", json.dumps(aapl_news, indent=2))
        else:
            print("未能获取AAPL的公司新闻 (函数返回None)。可能是API密钥问题、额度问题或无此股票新闻。")

        print("\n测试2: 获取一个不存在的股票代码 (e.g., 'NONEXISTENTTICKERXYZ') 的新闻，预期失败")
        non_existent_news = get_company_news_iex("NONEXISTENTTICKERXYZ", count=2)
        if non_existent_news is None:
            print("成功：对于不存在的股票代码，未获取到新闻 (符合预期)。")
        elif isinstance(non_existent_news, list) and not non_existent_news:
            print("成功：对于不存在的股票代码，返回空新闻列表 (符合预期)。")
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

    print("\n所有IEX Cloud测试结束。")
    print("注意: IEX Cloud的免费/沙箱API密钥返回的是混淆/示例数据。真实数据需要付费订阅。")
    print("API调用会消耗消息额度，请注意您的账户限制。") 