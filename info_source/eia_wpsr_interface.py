# Interface for accessing EIA WPSR data and other EIA energy data via their official API.
# analyze_tool.md mentions EIA's API for petroleum data.

import requests
import json
from datetime import datetime, timedelta
import os # Added for os.environ.get
from dotenv import load_dotenv # Added for .env loading

load_dotenv() # Load environment variables from .env file

# API key is now primarily loaded from .env, with a fallback to the placeholder.
EIA_API_KEY = os.environ.get("EIA_API_KEY") or "YOUR_API_KEY_HERE"
EIA_API_V2_BASE_URL = "https://api.eia.gov/v2"

def get_eia_series_data(series_id, api_key=None, start_date_str=None, end_date_str=None, frequency="weekly", data_columns=None):
    """
    Fetches data for a specific EIA series ID using API v2.

    Args:
        series_id (str): The EIA series ID (e.g., "PET.WCRSTUS1.W").
        api_key (str, optional): Your registered EIA API key. If None, uses the globally defined EIA_API_KEY.
        start_date_str (str, optional): Start date in "YYYY-MM-DD" format. Defaults to None.
        end_date_str (str, optional): End date in "YYYY-MM-DD" format. Defaults to None.
        frequency (str, optional): Data frequency. Examples: "weekly", "monthly", "annual". Defaults to "weekly".
        data_columns (list, optional): Specific data columns to retrieve, e.g., ["value"]. Defaults to ["value"].

    Returns:
        dict: JSON response from the API, or None if an error occurs or API key is missing.
              The actual data points are typically in response_dict[\'response\'][\'data\'].
    """
    current_api_key = api_key or EIA_API_KEY
    if current_api_key == "YOUR_API_KEY_HERE" or current_api_key is None:
        print("错误：EIA_API_KEY 未配置。请在 .env 文件中或直接在脚本中设置。")
        print("请访问 https://www.eia.gov/opendata/ 获取API密钥。")
        return None

    if data_columns is None:
        data_columns = ["value"]

    url = f"{EIA_API_V2_BASE_URL}/seriesid/{series_id}/data"

    params = {
        "api_key": current_api_key,
        "frequency": frequency,
    }
    # EIA API expects data columns in the format data[0]=value, data[1]=another_column if multiple are requested.
    # For a single 'value', it's simpler as 'data[]=value'.
    # Using a list of tuples for params for 'data[]' to ensure correct formatting by requests library for multiple values if needed.
    # However, the API doc examples show data[0]=value, data[1]=value2 for multiple columns.
    # For simplicity here, we'll assume 'data_columns' is a list and construct the 'data[x]' params.
    
    for i, col_name in enumerate(data_columns):
        params[f"data[{i}]"] = col_name

    if start_date_str:
        params["start"] = start_date_str
    if end_date_str:
        params["end"] = end_date_str
    
    # For specific WPSR data, it might be more direct to use routes like:
    # /petroleum/stoc/wstk/data with facets for series_id if series_id is a facet.
    # e.g., params['facets[series][]'] = series_id (if series_id is PET.WCRSTUS1.W)
    # But the /seriesid/ route is more direct for a known series.

    response = None # Initialize response
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
        
        parsed_response = response.json()
        if 'response' not in parsed_response or 'data' not in parsed_response['response']:
            # Check for API error messages if standard data structure is not present
            if 'error' in parsed_response: # Common error structure for API v1 like behavior
                print(f"EIA API返回错误: {parsed_response['error']}")
            elif 'data' in parsed_response and 'error' in parsed_response['data']: # API v2 error structure
                 print(f"EIA API返回错误: {parsed_response['data']['error']}")
            else:
                print("EIA API响应格式不符合预期，缺少 'response' 或 'response.data' 键。")
            print(f"完整响应: {json.dumps(parsed_response, indent=2)}")
            return None
        return parsed_response

    except requests.exceptions.RequestException as e:
        print(f"调用EIA API时发生错误: {e}")
        if response is not None:
            print(f"响应状态码: {response.status_code}")
            print(f"响应内容: {response.text}")
        return None
    except json.JSONDecodeError:
        print("解析EIA API响应时发生JSON解码错误。")
        if response is not None:
            print(f"响应文本: {response.text}")
        return None

# --- 基础测试 ---
if __name__ == "__main__":
    print("--- 测试 EIA API ---")

    if EIA_API_KEY == "YOUR_API_KEY_HERE" or EIA_API_KEY is None:
        print("请先在 .env 文件或脚本中配置 EIA_API_KEY 再运行测试。")
        print("跳过EIA API的实时测试。")
    else:
        print("\n测试1: 获取美国每周原油库存 (PET.WCRSTUS1.W)")
        
        # 获取最近4周的数据
        today = datetime.today()
        end_date = today
        start_date = today - timedelta(weeks=5) # Fetch a bit more to ensure we get 4 data points

        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d") # EIA API is usually inclusive for end date

        print(f"请求时间范围: {start_date_str} 到 {end_date_str}")

        # WPSR data series ID for "Weekly U.S. Ending Stocks of Crude Oil"
        crude_stocks_series_id = "PET.WCRSTUS1.W"
        
        # The EIA API documentation for v2 specifies data columns as data[0], data[1] etc.
        # We are interested in the 'value' of the series.
        data = get_eia_series_data(
            series_id=crude_stocks_series_id,
            start_date_str=start_date_str,
            end_date_str=end_date_str,
            frequency="weekly",
            data_columns=['value'] # Requesting the 'value' field
        )

        if data and data.get('response') and data['response'].get('data'):
            print(f"获取到 {data['response'].get('total', 'N/A')} 条记录。")
            print("最近的原油库存数据 (千桶):")
            # Data is typically newest first if not sorted, but let's sort in Python for display
            # EIA v2 API supports sort[0][column]=period and sort[0][direction]=desc
            # For now, we'll sort the received data if necessary
            retrieved_data = data['response']['data']
            # EIA data values are strings, ensure to cast to float/int if doing calculations
            # Example: {'period': '2023-10-20', 'value': '419697', 'value-units': 'thousand barrels'}
            for record in sorted(retrieved_data, key=lambda x: x.get('period'), reverse=True)[:4]: # Display up to 4 recent
                period = record.get('period')
                value = record.get('value')
                unit = record.get('value-units', 'N/A') # Some series might have units in metadata
                if 'value-units' not in record and 'description' in data['response'].get('series', [{}])[0].get('units', ''):
                     # Fallback: Try to get units from series metadata if available and not in data points
                     unit = data['response'].get('series', [{}])[0].get('units', 'N/A')

                print(f"  日期: {period}, 库存: {value} ({unit})")
        else:
            print("未能获取或解析原油库存数据。")
            if data:
                print("API响应部分内容:")
                print(json.dumps(data, indent=2, ensure_ascii=False))

    print("\n所有EIA测试结束。") 