# Interface for MarineTraffic API (Alternative Data - Shipping - as mentioned in analyze_tool.md) 

# Interface for accessing MarineTraffic API data.
# analyze_tool.md mentions MarineTraffic for vessel tracking and port calls.

import requests
import os
import json
from datetime import datetime
from dotenv import load_dotenv # Added for .env loading

load_dotenv() # Load environment variables from .env file

# API key is now primarily loaded from .env, with a fallback to the placeholder.
MARINE_TRAFFIC_API_KEY = os.environ.get("MARINETRAFFIC_API_KEY") or "YOUR_API_KEY_HERE"
MARINE_TRAFFIC_API_BASE_URL = "https://services.marinetraffic.com/api"

def get_marinetraffic_client_session():
    """Creates a requests session for MarineTraffic API calls."""
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    return session

def get_single_vessel_positions(mmsi=None, imo=None, ship_id=None, api_key=None, session=None):
    """
    Fetches the latest reported position of a single vessel identified by MMSI, IMO, or SHIPID.
    Corresponds to MarineTraffic API: Vessel Historical Track (last known position if days=0 or not set)
    or more directly, a single position fetch if available through a dedicated endpoint 
    (referencing servicedocs.marinetraffic.com for specific single vessel position endpoint if different from historical).
    For simplicity, we'll aim for a single position or very recent history.

    Args:
        mmsi (str, optional): MMSI of the vessel.
        imo (str, optional): IMO of the vessel.
        ship_id (str, optional): MarineTraffic SHIPID of the vessel.
        api_key (str, optional): Your MarineTraffic API key. If None, uses the globally defined MARINE_TRAFFIC_API_KEY.
        session (requests.Session, optional): Existing session to use for the request.

    Returns:
        dict: JSON response from the API, or None if an error occurs.
    """
    current_api_key = api_key or MARINE_TRAFFIC_API_KEY
    if current_api_key == "YOUR_API_KEY_HERE" or current_api_key is None:
        print("错误：MARINETRAFFIC_API_KEY 未配置。请在 .env 文件中或直接在脚本中设置。")
        return None

    if not (mmsi or imo or ship_id):
        print("错误：必须提供 MMSI, IMO, 或 SHIPID 中的至少一个。")
        return None
    
    # Based on servicedocs.marinetraffic.com, /get_single_vessel_positions seems appropriate
    # However, older or different API versions might use /exportvessel
    # We will use the path that seems most current for single vessel latest position if available,
    # or a historical track limited to a very short period.
    
    # Using a common endpoint for vessel positions - might need adjustment based on exact API version and subscription.
    # The documentation mentions getSingleVesselPositions under Vessel Positions. We will use /exportvessel as per common examples.
    # API URL Path: /exportvessel/v:8/{api_key}/mmsi:{mmsi}/protocol:jsono/timespan:{timespan_minutes}
    # For a single latest position, timespan can be set to a small value e.g. 5 or 10 minutes.
    # Or, if getSingleVesselPositions is a direct endpoint:
    # /ais/get_single_vessel_positions/v:1/{api_key}/mmsi:{mmsi}/protocol:jsono (example structure)
    
    # Let's try /exportvessel first as it's a common one cited in examples and covers latest positions with small timespan.
    # We need to choose one identifier. MMSI is often preferred if available.
    identifier_param = ""
    if mmsi:
        identifier_param = f"mmsi:{mmsi}"
    elif imo:
        identifier_param = f"imo:{imo}"
    elif ship_id:
        identifier_param = f"shipid:{ship_id}" # Note: API might use SHIP_ID or similar.

    # Requesting data for a very short timespan to get the latest position(s)
    # A timespan of 0 might work for some APIs to mean latest, or 1 to 5 minutes.
    # The API docs state /getSingleVesselPositions directly. Let's model after that logic.
    # The documentation points to: /vessel_historical_track/get_single_vessel_positions for a more direct approach
    # Or /vessel_positions/get_single_vessel_positions
    # Let's use the structure seen in servicedocs for single vessel positions, assuming it's the most current.
    # Example from docs: GET /ais/get_single_vessel_positions/v:1/{api_key}/mmsi:{mmsi_number}/protocol:{protocol}
    # The base URL used in the Python library example on PyPI is often services.marinetraffic.com/api/exportvessel
    # Given the new servicedocs, let's align with its structure for getSingleVesselPositions
    # It seems the newer path might be under /vessel_positions/get_single_vessel_positions
    # For simplicity and common usage, many examples use /exportvessel. We will provide a structure for that
    # and note that it might need to be adapted to the specific API version/subscription the user has.

    api_path_template = "/exportvessel/v:8/{api_key}/{identifier}/protocol:jsono/timespan:10"
    api_path = api_path_template.format(api_key=current_api_key, identifier=identifier_param)
    url = f"{MARINE_TRAFFIC_API_BASE_URL}{api_path}"
    
    _session = session or get_marinetraffic_client_session()
    response = None
    try:
        response = _session.get(url, timeout=30)
        response.raise_for_status()
        # MarineTraffic API (jsono) often returns a list of lists. The first list contains headers, the second contains data.
        # For a single vessel, it might be a list containing one data list.
        # Example: [[COLUMNS...],[DATA_VALUES...]] or just [DATA_VALUES...] if simplified. Need to check actual response.
        # The `Marine-Traffic-API` library on PyPI suggests response.models for parsed data.
        # Directly, response.json() will give the raw JSON. MarineTraffic sometimes wraps data in an array.
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            # If it returns a list of lists for single vessel (header + data rows)
            # or just a list of data rows. We are interested in the actual data records.
            # A simple single position might be a list with one dict, or a list of lists (header, data row).
            # Assuming a successful call returns a list of position objects (dicts) directly or a list of lists.
            return data 
        else:
            print(f"MarineTraffic API returned unexpected data format or empty data for {identifier_param}.")
            print(f"Response: {data}")
            return None 

    except requests.exceptions.RequestException as e:
        print(f"调用 MarineTraffic API (get_single_vessel_positions) 时发生错误: {e}")
        if response is not None:
            print(f"响应内容: {response.text}")
        return None
    except json.JSONDecodeError:
        print(f"解析 MarineTraffic API 响应时发生JSON解码错误 for {url}")
        if response is not None:
            print(f"响应文本: {response.text}")
        return None

# --- 基础测试 ---
if __name__ == "__main__":
    print("--- 测试 MarineTraffic API ---")

    if MARINE_TRAFFIC_API_KEY == "YOUR_API_KEY_HERE" or MARINE_TRAFFIC_API_KEY is None:
        print("请先在 .env 文件或脚本中配置 MARINE_TRAFFIC_API_KEY 再运行测试。")
        print("跳过MarineTraffic API的实时测试。")
    else:
        print("\n测试1: 获取特定船舶的最新位置 (示例使用MMSI)")
        # 用户需要替换为他们感兴趣的有效MMSI, IMO 或 SHIPID
        # 例如: EVER GIVEN (IMO: 9811000, MMSI: 355135000)
        test_mmsi = "355135000" 
        
        print(f"正在获取MMSI {test_mmsi} 的船舶位置...")
        vessel_position_data = get_single_vessel_positions(mmsi=test_mmsi)

        if vessel_position_data:
            print(f"成功获取到MMSI {test_mmsi} 的位置数据:")
            # The structure of the response needs to be handled based on MarineTraffic's specific API format.
            # If it's a list of lists (header, data): vessel_position_data[0] is header, vessel_position_data[1:] are data rows.
            # If it's a list of dicts (records), iterate directly.
            # Assuming jsono protocol with /exportvessel often returns [[HEADERS], [DATA_ROW_1_AS_LIST], [DATA_ROW_2_AS_LIST], ...]
            # For a single vessel and short timespan, we expect one data row.
            if isinstance(vessel_position_data, list) and len(vessel_position_data) > 1 and isinstance(vessel_position_data[0], list) and isinstance(vessel_position_data[1], list):
                headers = vessel_position_data[0]
                data_row = vessel_position_data[1]
                position_dict = dict(zip(headers, data_row))
                print(json.dumps(position_dict, indent=2, ensure_ascii=False))
            elif isinstance(vessel_position_data, list) and len(vessel_position_data) == 1 and isinstance(vessel_position_data[0], dict):
                 # If it returns a list with a single dictionary object
                 print(json.dumps(vessel_position_data[0], indent=2, ensure_ascii=False))
            else:
                print("收到的数据格式与预期不符，直接打印原始数据:")
                print(json.dumps(vessel_position_data, indent=2, ensure_ascii=False))
        else:
            print(f"未能获取MMSI {test_mmsi} 的位置数据。")

    print("\nMarineTraffic测试结束。")
    print("注意: MarineTraffic API的免费访问可能受限，高级功能通常需要付费订阅。确保您的API密钥有效并具有所需权限。") 