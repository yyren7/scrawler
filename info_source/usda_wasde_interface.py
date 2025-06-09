# Interface for accessing USDA WASDE data (e.g., via official API as mentioned in analyze_tool.md) 

# Interface for accessing USDA agricultural data, primarily via NASS QuickStats API
# WASDE report data is often available as a consolidated CSV (see get_wasde_report_csv function)
# analyze_tool.md mentions official API for USDA data. NASS QuickStats is a key one.

import requests
import json
import os # Added for os.environ.get
from dotenv import load_dotenv # Added for .env loading

load_dotenv() # Load environment variables from .env file

# API key is now primarily loaded from .env, with a fallback to the placeholder.
NASS_API_KEY = os.environ.get("NASS_API_KEY") or "YOUR_API_KEY_HERE"
NASS_API_BASE_URL = "https://quickstats.nass.usda.gov/api"

def get_nass_data(params):
    """
    Fetches data from the NASS QuickStats API.

    Args:
        params (dict): A dictionary of parameters for the API query.
                       Refer to NASS API documentation for available parameters:
                       https://quickstats.nass.usda.gov/api
                       Example: {'commodity_desc': 'CORN', 'year__GE': '2022', 'state_alpha': 'VA'}

    Returns:
        dict: JSON response from the API, or None if an error occurs or API key is missing.
    """
    if NASS_API_KEY == "YOUR_API_KEY_HERE":
        print("错误：NASS_API_KEY 未配置。请在 .env 文件中或直接在脚本中设置。")
        print("请访问 https://quickstats.nass.usda.gov/api 获取API密钥。")
        return None

    all_params = params.copy()
    all_params['key'] = NASS_API_KEY
    all_params['format'] = 'JSON' #  Can also be CSV or XML

    response = None # Initialize response variable
    try:
        response = requests.get(f"{NASS_API_BASE_URL}/api_GET/", params=all_params)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"调用NASS API时发生错误: {e}")
        if response is not None:
            print(f"响应内容: {response.text}")
        return None
    except json.JSONDecodeError:
        print("解析NASS API响应时发生JSON解码错误。")
        if response is not None: # Check if response object exists
            print(f"响应文本: {response.text}")
        return None

def get_nass_param_values(param_name):
    """
    Fetches possible values for a given NASS API parameter.

    Args:
        param_name (str): The name of the parameter (e.g., 'sector_desc', 'commodity_desc').

    Returns:
        dict: JSON response containing the parameter values, or None if an error occurs or API key is missing.
    """
    if NASS_API_KEY == "YOUR_API_KEY_HERE":
        print("错误：NASS_API_KEY 未配置。请在 .env 文件中或直接在脚本中设置。")
        return None

    params = {
        'key': NASS_API_KEY,
        'param': param_name
    }
    response = None # Initialize response variable
    try:
        response = requests.get(f"{NASS_API_BASE_URL}/get_param_values/", params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"调用NASS API (get_param_values) 时发生错误: {e}")
        return None
    except json.JSONDecodeError:
        print("解析NASS API响应时发生JSON解码错误 (get_param_values)。")
        if response is not None: # Check if response object exists
             print(f"响应文本: {response.text}")
        return None

def get_wasde_report_csv(url, save_path="wasde_latest.csv"):
    """
    Downloads the WASDE report CSV from the given URL.
    The primary URL for consolidated WASDE data is often found on:
    https://www.usda.gov/oce/commodity-markets/wasde-report

    Args:
        url (str): The direct URL to the WASDE CSV file.
        save_path (str): Path to save the downloaded CSV file.

    Returns:
        bool: True if download was successful, False otherwise.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"WASDE CSV报告已成功下载到 {save_path}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"下载WASDE CSV报告时发生错误: {e}")
        return False

# --- 基础测试 ---
if __name__ == "__main__":
    print("--- 测试 NASS QuickStats API ---")
    # 提醒用户配置API密钥
    if NASS_API_KEY == "YOUR_API_KEY_HERE":
        print("请先在 .env 文件或脚本中配置 NASS_API_KEY 再运行测试。")
        print("跳过NASS API的实时测试。")
    else:
        print("\\n测试1: 获取 'sector_desc' 参数的可能值")
        sector_values = get_nass_param_values('sector_desc')
        if sector_values:
            print("sector_desc 的可能值:", json.dumps(sector_values, indent=2))

        print("\\n测试2: 获取弗吉尼亚州（VA）2022年及以后玉米种植面积数据")
        # 示例: 获取弗吉尼亚州(VA)玉米(CORN)的种植面积(AREA PLANTED)数据，从2022年开始
        # 'AREA PLANTED' 对应的 'statisticcat_desc'
        # 短描述 'CORN - ACRES PLANTED'
        corn_params = {
            'source_desc': 'SURVEY', #来源描述
            'sector_desc': 'CROPS', # 部门描述：作物
            'group_desc': 'FIELD CROPS', # 分组描述：大田作物
            'commodity_desc': 'CORN', # 商品描述：玉米
            'statisticcat_desc': 'AREA PLANTED', # 统计类别描述：种植面积
            'agg_level_desc': 'STATE', # 聚合级别描述：州
            'year__GE': '2022', # 年份大于等于2022
            'state_alpha': 'VA' # 州缩写：弗吉尼亚州
            # 'short_desc': 'CORN - ACRES PLANTED' # 可以用short_desc简化查询，但需要确保准确
        }
        corn_data = get_nass_data(corn_params)
        if corn_data and 'data' in corn_data:
            print(f"获取到 {len(corn_data['data'])} 条关于弗吉尼亚州玉米的数据:")
            for record in corn_data['data'][:3]: # 打印前3条记录
                print(json.dumps(record, indent=2, ensure_ascii=False))
        elif corn_data:
            print("获取玉米数据成功，但响应中没有'data'字段或数据为空。")
            print("API 响应:", json.dumps(corn_data, indent=2))
        else:
            print("获取玉米数据失败。")

    print("\\n--- 测试 WASDE CSV 报告下载 ---")
    # 注意：这个URL需要是直接指向CSV文件的链接。
    # 用户需要从 https://www.usda.gov/oce/commodity-markets/wasde-report 页面找到最新的CSV链接。
    # 例如: "Consolidated Historical WASDE Report Data"
    # 此处使用一个占位符URL，因为实际URL可能每月变化。
    # 最新(截至2024年中)的链接格式通常类似:
    # https://www.usda.gov/sites/default/files/documents/Consolidated_Hist_WASDE_Full.csv 
    # 或 https://www.usda.gov/oce/commodity/wasde/latest.csv (这个似乎不总是最新CSV)
    # 更好的办法是访问WASDE主页查看"Consolidated Historical WASDE Report Data"的链接
    
    wasde_csv_url_placeholder = "PLACEHOLDER_URL_FOR_WASDE_CSV" 
    # 一个存档的示例URL (可能不是最新的):
    # wasde_csv_url_example = "https://www.usda.gov/sites/default/files/documents/Consolidated_Hist_WASDE_Jun24.csv"
    
    print(f"请访问 https://www.usda.gov/oce/commodity-markets/wasde-report 获取最新的WASDE CSV报告下载链接。")
    print(f"然后更新脚本中的 'wasde_csv_url_placeholder' 变量以进行实际测试。")
    print(f"当前占位符 URL: {wasde_csv_url_placeholder}")

    # 为了演示，这里不实际下载，因为URL是占位符
    # if wasde_csv_url_placeholder != "PLACEHOLDER_URL_FOR_WASDE_CSV":
    #     print(f"\\n尝试从 {wasde_csv_url_placeholder} 下载 WASDE CSV...")
    #     download_success = get_wasde_report_csv(wasde_csv_url_placeholder, "wasde_test.csv")
    #     if download_success:
    #         print("下载测试完成。请检查 wasde_test.csv 文件。")
    #         # 可以添加代码来读取并打印CSV的前几行
    #         try:
    #             import pandas as pd
    #             df = pd.read_csv("wasde_test.csv", nrows=5)
    #             print("WASDE CSV 文件的前5行:")
    #             print(df)
    #         except ImportError:
    #             print("未安装pandas库，无法预览CSV内容。可手动打开文件查看。")
    #         except Exception as e:
    #             print(f"读取CSV文件时发生错误: {e}")
    # else:
    #     print("\\n跳过WASDE CSV下载测试，因为URL是占位符。")
    print("所有测试结束。") 