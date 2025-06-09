# Interface for accessing Dalian Commodity Exchange (DCE) data.
# analyze_tool.md mentions DCE for daily prices and market statistics.
# This interface attempts to fetch data by emulating the 'Export Text/Excel' 
# functionality from the DCE public website.

import requests
import pandas as pd
from io import StringIO, BytesIO
from datetime import datetime

DCE_EXPORT_URL = "http://www.dce.com.cn/publicweb/quotesdata/exportDayQuotesFetch.html"

# Common DCE commodity codes (these might need verification or updates)
# Extracted from looking at DCE website's HTML or common knowledge.
# The GitHub repo yuany3721/DCEData was also a reference.
DCE_COMMODITY_CODES = {
    "ALL": "all",
    "SOYBEAN_NO1": "a",  # 豆一
    "SOYBEAN_NO2": "b",  # 豆二
    "SOYBEAN_MEAL": "m", # 豆粕
    "SOYBEAN_OIL": "y",  # 豆油
    "PALM_OLEIN": "p",   # 棕榈油
    "CORN": "c",         # 玉米
    "CORN_STARCH": "cs", # 玉米淀粉
    "EGG": "jd",         # 鸡蛋
    "IRON_ORE": "i",     # 铁矿石
    "COKING_COAL": "jm", # 焦煤
    "COKE": "j",         # 焦炭
    "LLDPE": "l",        # 聚乙烯 (Linear Low-Density Polyethylene)
    "PVC": "v",          # 聚氯乙烯 (Polyvinyl Chloride)
    "POLYPROPYLENE": "pp", # 聚丙烯
    "ETHYLENE_GLYCOL": "eg", # 乙二醇
    "STYRENE_MONOMER": "eb", # 苯乙烯
    "LPG": "pg",         # 液化石油气
    # Add more as needed
}

def get_dce_daily_data(date_str, commodity_code="all", trade_type='futures', export_format='txt'):
    """
    Fetches daily market data from Dalian Commodity Exchange (DCE).

    Args:
        date_str (str): Date for the data in "YYYYMMDD" format.
        commodity_code (str, optional): DCE commodity code (e.g., 'i' for iron ore, 'm' for soybean meal).
                                      Defaults to "all". Use keys from DCE_COMMODITY_CODES or their values.
        trade_type (str, optional): Type of trade. 'futures' (default) or 'options'.
        export_format (str, optional): 'txt' (default) or 'excel'. 'txt' is generally easier to parse.

    Returns:
        pandas.DataFrame: A DataFrame containing the daily data, or None if an error occurs.
    """
    try:
        dt_object = datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        print(f"日期格式错误: {date_str}。请使用 'YYYYMMDD' 格式。")
        return None

    # DCE website uses 0-indexed month for POST parameters
    post_data = {
        "dayQuotes.variety": commodity_code if commodity_code != "all" else "", # 'all' should be empty string for variety
        "dayQuotes.trade_type": "0" if trade_type == 'futures' else "1", # 0 for Futures, 1 for Options
        "year": str(dt_object.year),
        "month": str(dt_object.month - 1), # 0-indexed month
        "day": str(dt_object.day),
        "exportFlag": export_format
    }
    
    # It's good practice to use headers to mimic a browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Referer": "http://www.dce.com.cn/publicweb/quotesdata/dayQuotesEn.html", # Or the Chinese version
        "Content-Type": "application/x-www-form-urlencoded"
    }

    try:
        print(f"正在从DCE获取数据，参数: {post_data}")
        response = requests.post(DCE_EXPORT_URL, data=post_data, headers=headers, timeout=20)
        response.raise_for_status()

        if export_format == 'txt':
            # The text file from DCE is often in GBK or GB2312 encoding.
            # Try to determine encoding, default to gbk if unsure.
            content_text = ""
            try:
                content_text = response.content.decode('gbk')
            except UnicodeDecodeError:
                try:
                    content_text = response.content.decode('gb2312')
                except UnicodeDecodeError:
                    print("使用GBK和GB2312解码失败，尝试使用UTF-8。")
                    content_text = response.text # Fallback to requests' auto-detected encoding (often UTF-8)
            
            if not content_text.strip() or "No Data" in content_text or "没有查询到任何数据" in content_text:
                print(f"DCE在日期 {date_str} 没有找到品种 {commodity_code} 的数据。")
                if len(content_text) < 200 : print(f"响应内容: {content_text}")
                return None
            
            # The TXT file might have a header row and then data.
            # It's often tab-separated or has multiple spaces acting as delimiters.
            # We'll try common delimiters.
            try:
                # Skip rows until we find a line that looks like a header or data
                lines = content_text.splitlines()
                data_start_line = 0
                for i, line in enumerate(lines):
                    if "Contract" in line or "合约" in line or (line.strip() and len(line.split()) > 5): # Heuristic for header/data
                        data_start_line = i
                        break
                
                relevant_text = "\n".join(lines[data_start_line:])
                
                # Try reading with tab delimiter
                df = pd.read_csv(StringIO(relevant_text), sep='\t', engine='python')
                if df.shape[1] < 5: # If tab didn't work well, try multiple spaces
                    df = pd.read_csv(StringIO(relevant_text), delim_whitespace=True, engine='python')

            except Exception as parse_err:
                print(f"解析DCE TXT数据时发生错误: {parse_err}")
                print("原始文本数据 (前500字符):\n", content_text[:500])
                return None
            
            # Clean up DataFrame: remove rows that are all NaN (often an issue with parsed text)
            df.dropna(how='all', inplace=True)
            # Check if the first row looks like a header and set it if pandas didn't auto-detect
            if not df.empty and isinstance(df.columns[0], int) and ("Contract" in df.iloc[0,0] or "合约" in df.iloc[0,0]):
                 df.columns = df.iloc[0].str.strip()
                 df = df[1:].reset_index(drop=True)

            return df

        elif export_format == 'excel':
            # For Excel, pandas can read directly from bytes
            if not response.content:
                print(f"DCE在日期 {date_str} 没有找到品种 {commodity_code} 的数据 (Excel格式为空)。")
                return None
            try:
                # DCE Excel files might be .xls (Excel 97-2003) or .xlsx
                # pandas read_excel can often auto-detect. Try 'xlrd' for .xls, 'openpyxl' for .xlsx
                try:
                    df = pd.read_excel(BytesIO(response.content), engine='xlrd') # For .xls
                except Exception: # Fallback to openpyxl
                    df = pd.read_excel(BytesIO(response.content), engine='openpyxl') # For .xlsx
                return df
            except Exception as e:
                print(f"解析DCE Excel数据时发生错误: {e}")
                print("请确保已安装 'xlrd' 和 'openpyxl' 库: pip install xlrd openpyxl")
                # You might want to save the content to a file for manual inspection
                # with open(f"dce_error_{date_str}.xls", "wb") as f:
                #     f.write(response.content)
                # print("Excel内容已保存到 dce_error_{date_str}.xls 以供检查。")
                return None
        else:
            print(f"不支持的导出格式: {export_format}")
            return None

    except requests.exceptions.Timeout:
        print(f"从DCE获取数据超时。URL: {DCE_EXPORT_URL}, Data: {post_data}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"从DCE获取数据时发生网络错误: {e}")
        return None
    except Exception as e:
        print(f"处理DCE数据时发生未知错误: {e}")
        return None

# --- 基础测试 ---
if __name__ == "__main__":
    print("--- 测试大连商品交易所 (DCE) 数据接口 ---")

    # 获取昨天的日期作为测试 (交易所通常当天数据在收盘结算后才完整)
    # For testing, you might want to use a known past date where data exists.
    # today = datetime.now()
    # yesterday = today - timedelta(days=1)
    # test_date_str = yesterday.strftime("%Y%m%d")
    
    # 使用一个固定的过去日期进行测试，以确保数据存在性（如果当天是周末/假日则可能无数据）
    test_date_str = "20240715" # 周一，假设有数据 (请根据实际情况调整)
    print(f"\n测试日期: {test_date_str}")

    print(f"\n测试1: 获取 {test_date_str} 所有品种的期货数据 (TXT格式)")
    all_futures_txt = get_dce_daily_data(test_date_str, commodity_code="all", trade_type='futures', export_format='txt')
    if all_futures_txt is not None and not all_futures_txt.empty:
        print(f"成功获取到 {len(all_futures_txt)} 条记录。")
        print("所有品种期货数据 (TXT, 前5行):")
        print(all_futures_txt.head())
    elif all_futures_txt is not None and all_futures_txt.empty :
         print("获取到数据，但DataFrame为空。可能当日无此品种交易或解析问题。")
    else:
        print(f"未能获取 {test_date_str} 所有品种的期货数据 (TXT)。")

    print(f"\n测试2: 获取 {test_date_str} 铁矿石 ('i') 期货数据 (TXT格式)")
    iron_ore_code = DCE_COMMODITY_CODES["IRON_ORE"]
    iron_ore_futures_txt = get_dce_daily_data(test_date_str, commodity_code=iron_ore_code, trade_type='futures', export_format='txt')
    if iron_ore_futures_txt is not None and not iron_ore_futures_txt.empty:
        print(f"成功获取到 {len(iron_ore_futures_txt)} 条铁矿石记录。")
        print(f"铁矿石 ('{iron_ore_code}') 期货数据 (TXT, 前5行):")
        print(iron_ore_futures_txt.head())
    elif iron_ore_futures_txt is not None and iron_ore_futures_txt.empty:
         print("获取到数据，但DataFrame为空。可能当日无此品种交易或解析问题。")
    else:
        print(f"未能获取 {test_date_str} 铁矿石期货数据 (TXT)。")

    print(f"\n测试3: 获取 {test_date_str} 豆粕 ('m') 期货数据 (Excel格式)")
    soybean_meal_code = DCE_COMMODITY_CODES["SOYBEAN_MEAL"]
    # Note: Excel parsing requires 'xlrd' or 'openpyxl'. Install if not present.
    soybean_meal_futures_xls = get_dce_daily_data(test_date_str, commodity_code=soybean_meal_code, trade_type='futures', export_format='excel')
    if soybean_meal_futures_xls is not None and not soybean_meal_futures_xls.empty:
        print(f"成功获取到 {len(soybean_meal_futures_xls)} 条豆粕记录。")
        print(f"豆粕 ('{soybean_meal_code}') 期货数据 (Excel, 前5行):")
        print(soybean_meal_futures_xls.head())
    elif soybean_meal_futures_xls is not None and soybean_meal_futures_xls.empty :
         print("获取到数据，但DataFrame为空。可能当日无此品种交易或解析问题。")
    else:
        print(f"未能获取 {test_date_str} 豆粕期货数据 (Excel)。")
        print("如果Excel解析失败，请确保已安装 'xlrd' (for .xls) 和 'openpyxl' (for .xlsx) Python库。")
        print("例如: pip install xlrd openpyxl")

    # 测试一个通常有期权的品种，如豆粕
    print(f"\n测试4: 获取 {test_date_str} 豆粕 ('m') 期权数据 (TXT格式)")
    soybean_meal_options_txt = get_dce_daily_data(test_date_str, commodity_code=soybean_meal_code, trade_type='options', export_format='txt')
    if soybean_meal_options_txt is not None and not soybean_meal_options_txt.empty:
        print(f"成功获取到 {len(soybean_meal_options_txt)} 条豆粕期权记录。")
        print(f"豆粕 ('{soybean_meal_code}') 期权数据 (TXT, 前5行):")
        print(soybean_meal_options_txt.head())
    elif soybean_meal_options_txt is not None and soybean_meal_options_txt.empty:
        print("获取到数据，但DataFrame为空。可能当日无此品种期权交易或解析问题。")
    else:
        print(f"未能获取 {test_date_str} 豆粕期权数据 (TXT)。")
        
    print("\n所有DCE测试结束。") 