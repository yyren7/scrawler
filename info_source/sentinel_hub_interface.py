# Interface for Sentinel Hub API (Alternative Data - Satellite Imagery - as mentioned in analyze_tool.md) 

# Interface for accessing Sentinel Hub API data.
# analyze_tool.md mentions Sentinel Hub for satellite imagery.

import os
import json
from datetime import datetime, timedelta

from sentinelhub import (
    SentinelHubRequest, DataCollection, MimeType, CRS, BBox, SHConfig, \
    SentinelHubCatalog, Geometry, SentinelHubDownloadClient
)
from dotenv import load_dotenv # Added for .env loading

load_dotenv() # Load environment variables from .env file

# The sentinelhub-py library is the recommended way to interact with Sentinel Hub APIs.
# Users need to install it: pip install sentinelhub-py (or just sentinelhub)
# Configuration of Sentinel Hub credentials (client_id, client_secret) is typically done
# via a config file (~/.sentinelhub/config.ini) or environment variables.
# See: https://sentinelhub-py.readthedocs.io/en/latest/configure.html

# Global Sentinel Hub SHConfig object
sh_config_instance = None

def get_sh_config():
    """Initializes and returns a SHConfig object for Sentinel Hub API access."""
    global sh_config_instance
    if sh_config_instance is None:
        try:
            sh_config_instance = SHConfig()
            # If sentinelhub-py didn't find credentials in its default locations,
            # try to set them from environment variables loaded by dotenv.
            if not sh_config_instance.sh_client_id and os.environ.get("SH_CLIENT_ID"):
                sh_config_instance.sh_client_id = os.environ.get("SH_CLIENT_ID")
            if not sh_config_instance.sh_client_secret and os.environ.get("SH_CLIENT_SECRET"):
                sh_config_instance.sh_client_secret = os.environ.get("SH_CLIENT_SECRET")
            
            # Final check if credentials are set
            if not sh_config_instance.sh_client_id or not sh_config_instance.sh_client_secret:
                print("错误: Sentinel Hub Client ID 和 Client Secret 未配置。")
                print("请通过以下方式之一配置:")
                print("  1. 在项目根目录的 .env 文件中设置 SH_CLIENT_ID 和 SH_CLIENT_SECRET。")
                print("  2. 使用 `sentinelhub.config --sh_client_id YOUR_CLIENT_ID --sh_client_secret YOUR_CLIENT_SECRET` 命令。")
                print("  3. 创建或编辑 ~/.sentinelhub/config.ini 文件。")
                print("详细信息请参考: https://sentinelhub-py.readthedocs.io/en/latest/configure.html")
                sh_config_instance = None # Ensure it's None if not configured
        except Exception as e:
            print(f"初始化Sentinel Hub配置时发生错误: {e}")
            sh_config_instance = None
    return sh_config_instance

def search_sentinel_imagery(bbox_coords, time_interval, data_collection=DataCollection.SENTINEL2_L2A, cloud_cover_percent=None, config=None):
    """
    Searches for satellite imagery using Sentinel Hub Catalog API.

    Args:
        bbox_coords (list or tuple): Bounding box coordinates [min_lon, min_lat, max_lon, max_lat].
        time_interval (tuple or list): Tuple or list with start and end date strings (YYYY-MM-DD).
        data_collection (DataCollection, optional): The data collection to search.
                                                    Defaults to DataCollection.SENTINEL2_L2A.
        cloud_cover_percent (float, optional): Maximum cloud cover percentage (0-1, e.g., 0.6 for 60%).
                                                  Defaults to None (no cloud cover filter beyond collection defaults).
        config (SHConfig, optional): Sentinel Hub configuration object. If None, attempts to load default.

    Returns:
        list: A list of search results (features from Catalog API), or None if an error occurs.
    """
    active_config = config or get_sh_config()
    if not active_config or not active_config.sh_client_id:
        print("Sentinel Hub 配置无效或未提供，无法执行搜索。")
        return None

    try:
        catalog = SentinelHubCatalog(config=active_config)
        search_bbox = BBox(bbox=bbox_coords, crs=CRS.WGS84)
        
        # Constructing the filter for cloud cover using CQL2-TEXT if provided
        # Sentinel Hub Catalog API expects cloud cover as a percentage (0-100)
        # The property name for cloud cover is typically 'eo:cloud_cover'
        query_filter = None
        if cloud_cover_percent is not None:
            # Ensure cloud_cover_percent is correctly scaled if it's 0-1
            cc_value = int(cloud_cover_percent * 100) if cloud_cover_percent <= 1.0 else int(cloud_cover_percent)
            query_filter = f"eo:cloud_cover <= {cc_value}"

        search_iterator = catalog.search(
            collection=data_collection,
            bbox=search_bbox,
            time=time_interval,
            filter=query_filter, 
            filter_lang='cql2-text' if query_filter else None
        )
        results = list(search_iterator)
        return results
    except Exception as e:
        print(f"Sentinel Hub Catalog API 搜索时发生错误: {e}")
        return None

def get_sentinel_image(bbox_coords, time_interval, data_collection=DataCollection.SENTINEL2_L2A, 
                       width=512, height=None, evalscript=None, config=None, mime_type=MimeType.PNG):
    """
    Requests a satellite image using Sentinel Hub Process API.

    Args:
        bbox_coords (list or tuple): Bounding box coordinates [min_lon, min_lat, max_lon, max_lat].
        time_interval (tuple or list): Tuple or list with start and end date strings (YYYY-MM-DD), 
                                     or a single date string for a specific date.
        data_collection (DataCollection, optional): The data collection.
                                                    Defaults to DataCollection.SENTINEL2_L2A.
        width (int, optional): Width of the output image in pixels. Defaults to 512.
        height (int, optional): Height of the output image. If None, it's calculated based on bbox aspect ratio.
        evalscript (str, optional): Custom evalscript for processing. 
                                    If None, a default true-color script for Sentinel-2 is used.
        config (SHConfig, optional): Sentinel Hub configuration object.
        mime_type (MimeType, optional): The desired output format. Defaults to MimeType.PNG.

    Returns:
        list of numpy.ndarray or bytes: The image data (list of arrays if multiple responses, or bytes if single).
                                        None if an error occurs.
    """
    active_config = config or get_sh_config()
    if not active_config or not active_config.sh_client_id:
        print("Sentinel Hub 配置无效或未提供，无法获取影像。")
        return None

    if evalscript is None and data_collection == DataCollection.SENTINEL2_L2A:
        evalscript = """
            //VERSION=3
            function setup() {
                return {
                    input: ["B04", "B03", "B02"], // Red, Green, Blue bands for Sentinel-2
                    output: { bands: 3 }
                };
            }
            function evaluatePixel(sample) {
                return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02];
            }
        """
    elif evalscript is None:
        print(f"错误: 对于数据集合 {data_collection.value}，需要提供 evalscript。")
        return None

    try:
        request_bbox = BBox(bbox=bbox_coords, crs=CRS.WGS84)
        
        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=data_collection,
                    time_interval=time_interval,
                )
            ],
            responses=[
                SentinelHubRequest.output_response('default', mime_type)
            ],
            bbox=request_bbox,
            size=[width, height] if height else [width, None],
            config=active_config
        )
        # .get_data() returns a list of numpy arrays (one for each response defined in `responses`)
        image_data_list = request.get_data()
        return image_data_list
    except Exception as e:
        print(f"Sentinel Hub Process API 请求时发生错误: {e}")
        return None

# --- 基础测试 ---
if __name__ == "__main__":
    print("--- 测试 Sentinel Hub API ---")
    
    sh_config = get_sh_config()
    if not sh_config:
        print("Sentinel Hub 未正确配置，跳过实时API测试。")
    else:
        print(f"使用 Client ID: {sh_config.sh_client_id[:5]}... 进行测试 (如果已配置)")

        # Define a test Area of Interest (AOI) and time interval
        # Example: A small area in a well-known agricultural region (e.g., Central Valley, California)
        test_bbox_coords = [-121.5, 37.5, -121.0, 38.0] 
        
        # Time interval for catalog search (e.g., a month in the past)
        catalog_date_end_obj = datetime.now() - timedelta(days=60)
        catalog_date_start_obj = catalog_date_end_obj - timedelta(days=30)
        test_time_interval_str = (catalog_date_start_obj.strftime("%Y-%m-%d"), catalog_date_end_obj.strftime("%Y-%m-%d"))
        
        print(f"\n测试1: 使用Catalog API搜索 {DataCollection.SENTINEL2_L1C.value} 影像")
        print(f"AOI (bbox): {test_bbox_coords}")
        print(f"时间范围: {test_time_interval_str}")

        catalog_results = search_sentinel_imagery(
            bbox_coords=test_bbox_coords, 
            time_interval=test_time_interval_str,
            data_collection=DataCollection.SENTINEL2_L1C, 
            cloud_cover_percent=0.8 # Max 80% cloud cover
        )

        if catalog_results is not None:
            print(f"Catalog API 找到 {len(catalog_results)} 个结果。")
            if catalog_results:
                print("第一个结果的属性示例 (日期和云量):")
                first_result_props = catalog_results[0]['properties']
                print(f"  日期: {first_result_props.get('datetime')}")
                print(f"  云量 (eo:cloud_cover): {first_result_props.get('eo:cloud_cover')}")
        else:
            print("Catalog API 搜索失败或未配置。")

        print(f"\n测试2: 使用Process API获取 {DataCollection.SENTINEL2_L2A.value} 的小块影像 (真彩色)")
        # Use a more recent time for Process API, or a specific date from catalog results
        target_time_interval_for_process = test_time_interval_str # Default to catalog search interval
        if catalog_results and len(catalog_results) > 0:
            first_result_date = catalog_results[0]['properties']['datetime'].split('T')[0]
            # For Process API, often a narrower time range or specific date is better
            target_time_interval_for_process = (first_result_date, first_result_date) 
            print(f"将尝试使用 Catalog API 找到的日期 ({first_result_date}) 进行影像获取。")
        else:
            # If catalog search failed or returned no results, use a generic recent interval for L2A
            proc_date_end_obj = datetime.now() - timedelta(days=45) 
            proc_date_start_obj = proc_date_end_obj - timedelta(days=10)
            target_time_interval_for_process = (proc_date_start_obj.strftime("%Y-%m-%d"), proc_date_end_obj.strftime("%Y-%m-%d"))
            print(f"Catalog未找到结果，尝试通用时间范围: {target_time_interval_for_process}")

        image_data_list = get_sentinel_image(
            bbox_coords=test_bbox_coords, 
            time_interval=target_time_interval_for_process,
            data_collection=DataCollection.SENTINEL2_L2A,
            width=256 
        )

        if image_data_list and isinstance(image_data_list, list) and len(image_data_list) > 0:
            print(f"Process API 成功获取到影像数据。列表包含 {len(image_data_list)} 个元素。")
            print(f"第一个影像数据的类型: {type(image_data_list[0])}")
            if hasattr(image_data_list[0], 'shape'):
                 print(f"第一个影像数据的形状 (通常是 HxWxBands): {image_data_list[0].shape}")
            # To save the image (example for PNG):
            # try:
            #     from PIL import Image
            #     import numpy as np
            #     # Assuming the first item in the list is the image data (numpy array for PNG/JPEG)
            #     if isinstance(image_data_list[0], np.ndarray):
            #         img = Image.fromarray(image_data_list[0])
            #         img.save("sentinel_hub_test_image.png")
            #         print("示例影像已尝试保存为 sentinel_hub_test_image.png")
            # except ImportError:
            #     print("请安装Pillow库 (pip install Pillow) 来保存影像。")
            # except Exception as img_e:
            #     print(f"保存影像时出错: {img_e}")
        else:
            print("Process API未能获取影像数据或返回为空。")

    print("\n所有Sentinel Hub测试结束。")
    print("确保您的Sentinel Hub账户已正确配置 (client_id, client_secret)，并且有足够的处理单元 (PU) 来执行请求。")
    print("免费试用账户的PU数量有限。") 