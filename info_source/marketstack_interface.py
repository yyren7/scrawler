# Interface for accessing Marketstack API data.
# analyze_tool.md lists Marketstack under "Financial News APIs".
# However, based on public documentation (marketstack.com/documentation_v2),
# its primary offering is stock market data (EOD, intraday, splits, dividends, etc.).
# A dedicated endpoint for fetching news *articles* was not immediately apparent
# in the top-level API documentation found.
# This implementation will focus on a core Marketstack feature: End-of-Day stock data.

import requests
import json
from datetime import datetime
import os # Added for os.environ.get
from dotenv import load_dotenv # Added for .env loading

load_dotenv() # Load environment variables from .env file

# API key is now primarily loaded from .env, with a fallback to the placeholder.
MARKETSTACK_API_KEY = os.environ.get("MARKETSTACK_API_KEY") or "YOUR_ACCESS_KEY_HERE"
MARKETSTACK_API_V2_BASE_URL = "https://api.marketstack.com/v2" # Recommended version

def get_marketstack_eod_data(symbols, api_key=None, date_from=None, date_to=None, limit=100, exchange_mic=None):
    """
    Fetches End-of-Day (EOD) stock data from Marketstack API (V2).

    Args:
        symbols (str or list): A single stock symbol (e.g., "AAPL") or a list/comma-separated string 
                               of symbols (e.g., ["AAPL", "MSFT"] or "AAPL,MSFT").
        api_key (str, optional): Your Marketstack API access key. If None, uses the globally defined MARKETSTACK_API_KEY.
        date_from (str, optional): Start date in "YYYY-MM-DD" format. Defaults to None.
        date_to (str, optional): End date in "YYYY-MM-DD" format. Defaults to None.
        limit (int, optional): Number of data points to return. Max 1000 on some plans. Defaults to 100.
        exchange_mic (str, optional): Filter by a specific exchange MIC (e.g., "XNAS" for NASDAQ).
                                      Defaults to None.

    Returns:
        dict: JSON response from the API, or None if an error occurs or API key is missing.
              The actual data points are typically in response_dict[\'data\'].
    """
    current_api_key = api_key or MARKETSTACK_API_KEY
    if current_api_key == "YOUR_ACCESS_KEY_HERE" or current_api_key is None:
        print("错误：MARKETSTACK_API_KEY 未配置。请在 .env 文件中或直接在脚本中设置。")
        print("请访问 https://marketstack.com/dashboard 获取API密钥。")
        return None

    if isinstance(symbols, list):
        symbols_str = ",".join(symbols)
    else:
        symbols_str = symbols

    params = {
        "access_key": current_api_key,
        "symbols": symbols_str,
        "limit": limit
    }
    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to
    if exchange_mic:
        params["exchange"] = exchange_mic

    url = f"{MARKETSTACK_API_V2_BASE_URL}/eod"
    response = None

    try:
        response = requests.get(url, params=params)
        response.raise_for_status() 
        
        parsed_response = response.json()
        if 'error' in parsed_response:
            print(f"Marketstack API 返回错误: Code {parsed_response['error'].get('code')}, Message: {parsed_response['error'].get('message')}")
            if 'context' in parsed_response['error']:
                print(f"错误详情: {parsed_response['error']['context']}")
            return None
        if 'data' not in parsed_response:
            print("Marketstack API响应格式不符合预期，缺少 'data' 键。")
            print(f"完整响应: {json.dumps(parsed_response, indent=2)}")
            return None
            
        return parsed_response

    except requests.exceptions.RequestException as e:
        print(f"调用Marketstack API时发生错误: {e}")
        if response is not None:
            print(f"响应状态码: {response.status_code}")
            print(f"响应内容: {response.text}")
        return None
    except json.JSONDecodeError:
        print("解析Marketstack API响应时发生JSON解码错误。")
        if response is not None:
            print(f"响应文本: {response.text}")
        return None
    except Exception as e:
        print(f"处理Marketstack数据时发生未知错误: {e}")
        return None

# --- 基础测试 ---
if __name__ == "__main__":
    print("--- 测试 Marketstack API ---")

    if MARKETSTACK_API_KEY == "YOUR_ACCESS_KEY_HERE" or MARKETSTACK_API_KEY is None:
        print("请先在 .env 文件或脚本中配置 MARKETSTACK_API_KEY 再运行测试。")
        print("跳过Marketstack API的实时测试。")
    else:
        print("\n测试1: 获取 AAPL (Apple Inc.) 最近的日终数据 (EOD)")
        aapl_eod_data = get_marketstack_eod_data(symbols="AAPL", limit=5)

        if aapl_eod_data and aapl_eod_data.get('data'):
            print(f"获取到 {len(aapl_eod_data['data'])} 条 AAPL EOD 记录:")
            for record in aapl_eod_data['data']:
                print(f"  日期: {record.get('date')}, 收盘价: {record.get('close')}, 交易所: {record.get('exchange')}")
        elif aapl_eod_data:
            print("获取AAPL EOD数据成功，但响应中没有'data'字段或数据为空。")
            print("API 响应:", json.dumps(aapl_eod_data, indent=2))
        else:
            print("未能获取AAPL的EOD数据。")

        print("\n测试2: 获取 TSLA (Tesla Inc.) 在特定日期范围内的EOD数据")
        tsla_eod_data_range = get_marketstack_eod_data(
            symbols="TSLA", 
            date_from="2024-01-01", 
            date_to="2024-01-05",
            limit=5
        )

        if tsla_eod_data_range and tsla_eod_data_range.get('data'):
            print(f"获取到 {len(tsla_eod_data_range['data'])} 条 TSLA EOD 记录 (2024-01-01 to 2024-01-05):")
            for record in tsla_eod_data_range['data']:
                print(f"  日期: {record.get('date')}, 收盘价: {record.get('close')}, 成交量: {record.get('volume')}")
        elif tsla_eod_data_range:
             print("获取TSLA EOD数据成功，但响应中没有'data'字段或数据为空。")
             print("API 响应:", json.dumps(tsla_eod_data_range, indent=2))
        else:
            print("未能获取TSLA在指定日期范围内的EOD数据。")

    print("\n--- 关于Marketstack新闻数据的说明 ---")
    print("analyze_tool.md 将 Marketstack 列在金融新闻API下。")
    print("然而，其公开API文档 (V1和V2) 主要描述了市场数据 (股价、交易量、分红等) 的获取。")
    print("本实现基于其核心的市场数据功能。")
    print("如果您的 Marketstack 订阅计划包含专门的新闻文章API端点，请查阅您的订阅详情和完整文档，并可在此处补充相关函数。")

    print("\n所有Marketstack测试结束。") 