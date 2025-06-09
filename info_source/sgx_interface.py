# Interface for accessing Singapore Exchange (SGX) data.
# analyze_tool.md mentions SGX for iron ore derivatives data.
# This interface attempts to download daily historical data files (CSV)
# from the SGX website, based on observed URL patterns.

import requests
import pandas as pd
from io import StringIO
from datetime import datetime

SGX_DERIVATIVES_HISTORICAL_BASE_URL = "https://links.sgx.com/1.0.0/derivatives-historical"

# SGX Report Types (observed file prefixes)
SGX_REPORT_TYPE_TRADING_SUMMARY = "TC"  # Trading Summary (often Tick and Chart data, can be large)
SGX_REPORT_TYPE_SETTLEMENT_PRICES = "SD" # Settlement Prices (Closing prices, etc.)

def get_sgx_derivatives_daily_data(date_str, report_type="SD"):
    """
    Downloads and parses daily derivatives data from SGX.

    Args:
        date_str (str): Date for the data in "YYYYMMDD" format.
        report_type (str, optional): The type of report to download.
                                     "SD" for Settlement Prices (default).
                                     "TC" for Trading Summary.

    Returns:
        pandas.DataFrame: A DataFrame containing the daily data, or None if an error occurs.
    """
    try:
        # Validate date format, though not strictly needed for URL construction here
        datetime.strptime(date_str, "%Y%m%d") 
    except ValueError:
        print(f"日期格式错误: {date_str}。请使用 'YYYYMMDD' 格式。")
        return None

    if report_type not in [SGX_REPORT_TYPE_SETTLEMENT_PRICES, SGX_REPORT_TYPE_TRADING_SUMMARY]:
        print(f"不支持的报告类型: {report_type}。请使用 'SD' 或 'TC'。")
        return None

    file_name = f"{report_type}_{date_str}.csv"
    download_url = f"{SGX_DERIVATIVES_HISTORICAL_BASE_URL}/{date_str}/{file_name}"

    print(f"正在从SGX下载数据: {download_url}")

    response = None # Initialize response
    try:
        response = requests.get(download_url, timeout=30) # Increased timeout for potentially large files
        response.raise_for_status() # Will raise an HTTPError for bad status codes (4XX or 5XX)

        content_text = response.text
        if not content_text.strip():
            print(f"下载的文件为空: {download_url}")
            return None

        # Check for common SGX messages indicating no data or error page
        page_not_found_error = "The page you are looking for cannot be found" in content_text
        no_records_error = "No records found" in content_text
        is_html_error_page = response.headers.get('Content-Type', '').startswith('text/html') and \
                             ("error" in content_text.lower() or "cannot be found" in content_text.lower())

        if page_not_found_error or no_records_error or is_html_error_page:
            print(f"SGX在日期 {date_str} 没有找到报告 {report_type} 的数据，或者URL无效/返回错误页面。")
            print(f"URL: {download_url}")
            if len(content_text) < 500:
                 print(f"响应内容: {content_text}")
            return None

        df = pd.read_csv(StringIO(content_text))
        
        if df.empty:
            print(f"解析后的DataFrame为空。URL: {download_url}")
            return None
        
        if df.shape[1] <= 1 and len(content_text.splitlines()) > 1:
            print(f"CSV解析可能不正确 (只有一列数据)。请检查文件内容。URL: {download_url}")
            data_starts_after_keyword = "Contract Series"
            lines = content_text.splitlines()
            skiprows = 0
            found_keyword = False
            for i, line in enumerate(lines):
                if data_starts_after_keyword in line:
                    skiprows = i
                    found_keyword = True
                    break
            
            if found_keyword and skiprows > 0:
                print(f"尝试跳过 {skiprows} 行重新解析。")
                df = pd.read_csv(StringIO(content_text), skiprows=skiprows)
                if df.empty or df.shape[1] <=1:
                    print("跳行重新解析后仍然失败或数据列太少。")
                    return None
            else:
                 print("未找到关键词 '{data_starts_after_keyword}' 或解析仍然不佳。")
                 return None 

        print(f"成功下载和解析SGX报告 {report_type} for {date_str}。")
        return df

    except requests.exceptions.HTTPError as e:
        if response and e.response.status_code == 404:
            print(f"SGX报告未找到 (404错误): {download_url}")
        elif response:
            print(f"下载SGX数据时发生HTTP错误: {e} (URL: {download_url}, Status: {response.status_code})")
        else:
            print(f"下载SGX数据时发生HTTP错误 (无响应对象): {e} (URL: {download_url})")
        return None
    except requests.exceptions.Timeout:
        print(f"从SGX下载数据超时: {download_url}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"从SGX下载数据时发生网络错误: {e}")
        return None
    except pd.errors.EmptyDataError:
        print(f"Pandas解析错误: 无数据。下载的文件可能为空或不包含表格数据。URL: {download_url}")
        return None
    except Exception as e:
        print(f"处理SGX数据时发生未知错误: {e}")
        return None

# --- 基础测试 ---
if __name__ == "__main__":
    print("--- 测试新加坡交易所 (SGX) 数据接口 ---")

    test_date_str = "20240715" 
    print(f"\n测试日期: {test_date_str}")

    print(f"\n测试1: 获取 {test_date_str} 衍生品结算价格 (SD)")
    settlement_prices = get_sgx_derivatives_daily_data(test_date_str, report_type=SGX_REPORT_TYPE_SETTLEMENT_PRICES)
    if settlement_prices is not None and not settlement_prices.empty:
        print(f"成功获取到结算价格 {len(settlement_prices)} 条记录。")
        print("结算价格数据 (前5行):")
        print(settlement_prices.head())
        # 铁矿石相关合约的Symbol通常包含 'FEF', 'TSIF', 'IOC' 等
        # Ensure all data is string before applying .str.contains()
        iron_ore_contracts = settlement_prices[settlement_prices.astype(str).apply(lambda row: row.str.contains('FEF|TSIF|IOC', case=False, regex=True).any(), axis=1)]
        if not iron_ore_contracts.empty:
            print("\n找到可能的铁矿石相关合约 (示例):")
            print(iron_ore_contracts.head())
        else:
            print("\n在结算价格数据中未直接找到铁矿石关键词(FEF, TSIF, IOC)的合约。可能需要检查列名或合约命名规则。")
    elif settlement_prices is not None and settlement_prices.empty:
        print("获取到数据，但DataFrame为空。可能当日无交易或解析问题。")
    else:
        print(f"未能获取 {test_date_str} 的SGX结算价格数据。")

    print(f"\n测试2: 获取 {test_date_str} 衍生品交易摘要 (TC) - 文件可能较大")
    print("交易摘要(TC)文件下载测试默认被注释掉，因为文件可能很大，下载耗时较长。如需测试请取消注释。")

    print("\n--- SGX数据源说明 ---")
    print("此接口依赖于SGX网站上公开历史数据文件的URL模式和格式。")
    print("如果SGX更改其网站结构或URL模式，此脚本可能需要更新。")
    print("对于实时数据或通过专用API的访问，请参考SGX官方商业数据服务。")
    print("analyze_tool.md 中提到 Investing.com 等第三方平台也可能有SGX历史数据，但直接从交易所获取通常更可靠（如果可行）。")

    print("\n所有SGX测试结束。") 