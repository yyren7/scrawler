# Interface for accessing Twelve Data API.
# analyze_tool.md lists Twelve Data for broad financial market coverage.
# While its primary strength is market and fundamental data, not direct news articles,
# we will implement a function to fetch the Earnings Calendar, which is news-event related.

import os
import json
from datetime import datetime, timedelta
from twelvedata import TDClient # type: ignore
from dotenv import load_dotenv # Added for .env loading

load_dotenv() # Load environment variables from .env file

# API key is now primarily loaded from .env, with a fallback to the placeholder.
TWELVEDATA_API_KEY = os.environ.get("TWELVEDATA_API_KEY") or "YOUR_API_KEY_HERE"

# Global Twelve Data client instance
td_client_instance = None

def get_twelve_data_client():
    global td_client_instance
    if TWELVEDATA_API_KEY == "YOUR_API_KEY_HERE" or TWELVEDATA_API_KEY is None:
        print("错误：TWELVEDATA_API_KEY 未配置。请在 .env 文件中或直接在脚本中设置API密钥。")
        print("请访问 https://twelvedata.com/ 获取API密钥。")
        return None
    if td_client_instance is None:
        try:
            td_client_instance = TDClient(apikey=TWELVEDATA_API_KEY)
        except Exception as e:
            print(f"初始化Twelve Data客户端时发生错误: {e}")
            td_client_instance = None 
            return None
    return td_client_instance

def get_earnings_calendar(from_date_str, to_date_str, country=None, exchange=None, symbol=None, limit=100):
    """
    Fetches the earnings calendar from Twelve Data API.

    Args:
        from_date_str (str): Start date in "YYYY-MM-DD" format.
        to_date_str (str): End date in "YYYY-MM-DD" format.
        country (str, optional): Filter by country. Defaults to None.
        exchange (str, optional): Filter by exchange. Defaults to None.
        symbol (str, optional): Filter by specific symbol. Defaults to None.
        limit (int, optional): Number of results. Defaults to 100.

    Returns:
        dict: JSON-like response from the API (often a list of earnings events within a dict),
              or None if an error occurs.
              The actual earnings data is typically in response_dict[\'earnings\'].
    """
    td = get_twelve_data_client()
    if not td:
        return None

    try:
        # The official Python client `twelvedata` might wrap direct API calls.
        # We use the .get_earnings_calendar() method from the client.
        # Parameters need to be passed according to the client library's definition.
        # The library might convert these to appropriate API query params.
        calendar_request = td.get_earnings_calendar(
            date_from=from_date_str, 
            date_to=to_date_str,
            country=country,
            exchange=exchange,
            symbol=symbol,
            limit=limit
        )
        # The .as_json() method is usually available on the request object from the TDClient
        earnings_data = calendar_request.as_json()

        # Check for error messages in the response, as the library might not always raise exceptions
        # for API-level errors (e.g., plan limits, invalid parameters not caught by client).
        if isinstance(earnings_data, dict) and earnings_data.get("status") == "error":
            print(f"Twelve Data API (earnings_calendar) 返回错误: Code {earnings_data.get('code')}, Message: {earnings_data.get('message')}")
            return None
        
        # If no 'earnings' key but status is ok, it might be an empty result set which is valid.
        if isinstance(earnings_data, dict) and 'earnings' not in earnings_data and earnings_data.get("status") == "ok":
            print(f"在 {from_date_str} 到 {to_date_str} 期间未找到符合条件的财报日历。")
            # Return the original response as it might contain metadata or be an empty list wrapper.
            return earnings_data 

        return earnings_data
    except Exception as e:
        # This will catch errors from the client library itself (e.g., connection issues, unexpected responses)
        # or issues if .as_json() is called on None or an unexpected object type.
        print(f"调用Twelve Data API (get_earnings_calendar) 或处理响应时发生错误: {e}")
        return None

# --- 基础测试 ---
if __name__ == "__main__":
    print("--- 测试 Twelve Data API ---")

    client_init_success = bool(get_twelve_data_client()) 

    if not client_init_success:
        print("由于API密钥未配置或客户端初始化失败，跳过Twelve Data API的实时测试。")
    else:
        today = datetime.now()
        one_week_from_now = today + timedelta(days=7)
        
        from_date_test = today.strftime("%Y-%m-%d")
        to_date_test = one_week_from_now.strftime("%Y-%m-%d")

        print(f"\n测试1: 获取从 {from_date_test} 到 {to_date_test} 的财报日历 (美国)")
        earnings_us = get_earnings_calendar(from_date_str=from_date_test, to_date_str=to_date_test, country="United States", limit=5)

        if earnings_us and isinstance(earnings_us.get('earnings'), list):
            print(f"获取到 {len(earnings_us['earnings'])} 条美国公司财报日历记录:")
            for event in earnings_us['earnings'][:5]: # Display first 5
                print(f"  公司: {event.get('name')} ({event.get('symbol')}), 日期: {event.get('date')}, 时间: {event.get('time')}, EPS预期: {event.get('eps_estimate')}")
        elif earnings_us and earnings_us.get("status") == "ok":
            print(f"在 {from_date_test} 到 {to_date_test} 期间未找到美国公司财报日历记录。")
            # print("API 响应:", json.dumps(earnings_us, indent=2))
        else:
            print(f"未能获取美国公司财报日历数据。 API响应: {earnings_us}")

        print(f"\n测试2: 获取 AAPL 在 {from_date_test} 到 {to_date_test} 的财报信息 (如果有)")
        aapl_earnings = get_earnings_calendar(from_date_str=from_date_test, to_date_str=to_date_test, symbol="AAPL")
        if aapl_earnings and isinstance(aapl_earnings.get('earnings'), list) and aapl_earnings['earnings']:
            print(f"获取到AAPL的财报信息:")
            for event in aapl_earnings['earnings']:
                print(f"  公司: {event.get('name')} ({event.get('symbol')}), 日期: {event.get('date')}, 时间: {event.get('time')}")
        elif aapl_earnings and isinstance(aapl_earnings.get('earnings'), list) and not aapl_earnings['earnings']:
            print(f"在 {from_date_test} 到 {to_date_test} 期间未找到AAPL的财报信息。")
        elif aapl_earnings and aapl_earnings.get("status") == "ok":
             print(f"在 {from_date_test} 到 {to_date_test} 期间未找到AAPL的财报信息。")
             # print("API 响应:", json.dumps(aapl_earnings, indent=2))
        else:
            print(f"未能获取AAPL的财报信息。API响应: {aapl_earnings}")

    print("\n--- 关于Twelve Data新闻数据的说明 ---")
    print("analyze_tool.md 将 Twelve Data 列在金融新闻API下。")
    print("然而，其API文档主要展示了市场数据、基本面数据和技术指标等功能。")
    print("本实现选取了'财报日历'功能，因为它与新闻事件紧密相关。")
    print("Twelve Data 可能通过其他方式或特定合作提供纯新闻内容，但其核心API似乎更侧重结构化数据。")

    print("\n所有Twelve Data测试结束。")
    print("注意: Twelve Data的免费API密钥 (Basic Plan) 有请求频率和总量的限制。") 