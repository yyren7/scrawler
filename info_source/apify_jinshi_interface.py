# Interface for accessing Jinshi (金十) data via an Apify Actor.
# analyze_tool.md mentions Apify for Jinshi data, a Chinese financial news source.

import os
import json
from apify_client import ApifyClient # type: ignore
from dotenv import load_dotenv # Added for .env loading

load_dotenv() # Load environment variables from .env file

# API Token is now primarily loaded from .env, with a fallback to the placeholder.
APIFY_API_TOKEN = os.environ.get("APIFY_TOKEN") or "YOUR_APIFY_TOKEN_HERE"
JINSHI_NEWS_ACTOR_ID = "dadaodb/jinshi-news"

# Global Apify client instance
apify_client_instance = None

def get_apify_client():
    global apify_client_instance
    if APIFY_API_TOKEN == "YOUR_APIFY_TOKEN_HERE" or APIFY_API_TOKEN is None:
        print("错误：APIFY_TOKEN 未配置。请在 .env 文件中或直接在脚本中设置Apify API Token。")
        print("请访问 https://console.apify.com/account/integrations 获取API Token。")
        return None
    if apify_client_instance is None:
        try:
            apify_client_instance = ApifyClient(APIFY_API_TOKEN)
        except Exception as e:
            print(f"初始化Apify客户端时发生错误: {e}")
            apify_client_instance = None
            return None
    return apify_client_instance

def get_jinshi_news_via_apify(actor_input=None, wait_for_finish_secs=120):
    """
    Runs the Jinshi News Actor on Apify and fetches the results.

    Args:
        actor_input (dict, optional): Input for the Actor. For dadaodb/jinshi-news,
                                      it might accept parameters to filter news (e.g., date, type).
                                      Defaults to {} which likely fetches latest/default news.
                                      Consult the Actor's documentation on Apify for specific input schema.
        wait_for_finish_secs (int, optional): Maximum time to wait for the Actor run to finish.

    Returns:
        list: A list of news items (dictionaries) from the Actor's default dataset, or None if an error occurs.
    """
    client = get_apify_client()
    if not client:
        return None

    if actor_input is None:
        actor_input = {} # Default input for the Jinshi actor if not specified

    print(f"正在运行 Apify Actor: {JINSHI_NEWS_ACTOR_ID}，输入: {actor_input}")
    
    try:
        actor_run = client.actor(JINSHI_NEWS_ACTOR_ID).call(run_input=actor_input, wait_for_finish=wait_for_finish_secs)
        
        if not actor_run:
            print("Apify Actor .call() 未返回有效的运行对象。")
            return None
            
        run_id = actor_run.get("id")
        dataset_id = actor_run.get("defaultDatasetId")

        if not dataset_id:
            print(f"Apify Actor 运行 {run_id} 未找到默认数据集ID。运行状态: {actor_run.get('status')}")
            # You might want to check actor_run["status"] for details like "TIMED-OUT", "FAILED"
            return None

        print(f"Apify Actor 运行完成。Run ID: {run_id}, Dataset ID: {dataset_id}")
        print(f"可以在以下链接查看数据集: https://console.apify.com/storage/datasets/{dataset_id}")

        items = []
        # Iterate through the dataset items
        # The free plan might have limits on the number of results retrievable or total compute units.
        for item in client.dataset(dataset_id).iterate_items():
            items.append(item)
        
        if not items:
            print("Actor运行成功，但数据集为空。可能当前没有新的金十快讯，或Actor的抓取逻辑未返回数据。")

        return items

    except Exception as e:
        print(f"运行Apify Actor ({JINSHI_NEWS_ACTOR_ID}) 或获取结果时发生错误: {e}")
        # If it's a timeout from wait_for_finish, actor_run might be None or incomplete.
        # If the error is an API error from Apify, it might be caught here.
        return None

# --- 基础测试 ---
if __name__ == "__main__":
    print("--- 测试 Apify Jinshi News Actor 接口 ---")

    client_init_success = bool(get_apify_client())

    if not client_init_success:
        print("由于Apify API Token未配置或客户端初始化失败，跳过Apify Actor的实时测试。")
    else:
        print(f"使用的Apify API Token: {'已配置' if APIFY_API_TOKEN != 'YOUR_APIFY_TOKEN_HERE' else '未配置或为占位符'}")
        print(f"将要调用的Actor: {JINSHI_NEWS_ACTOR_ID}")
        print("注意: 运行Apify Actors会消耗您的Apify平台资源 (计算单元、代理等) 并可能产生费用。")
        print(f"此特定Actor ({JINSHI_NEWS_ACTOR_ID}) 的定价为每1000结果 $5.00。")

        print("\n测试1: 运行 Jinshi News Actor 并获取少量最新快讯 (默认输入)")
        # The default input for this actor is often empty, meaning it gets the latest news.
        # Consult the specific actor's documentation on Apify for input schema if you need to pass params.
        jinshi_news = get_jinshi_news_via_apify(wait_for_finish_secs=180) # Increased wait time

        if jinshi_news is not None:
            print(f"成功从Apify Actor获取到 {len(jinshi_news)} 条金十数据。")
            if jinshi_news:
                print("显示最多3条记录:")
                for i, news_item in enumerate(jinshi_news[:3]):
                    print(f"--- 新闻 {i+1} ---")
                    # The structure of `news_item` depends on what the Apify actor returns.
                    # Based on common Jinshi data, it might have 'time', 'content', 'importance' etc.
                    print(json.dumps(news_item, indent=2, ensure_ascii=False))
            else:
                print("数据集为空，但Actor运行可能已成功 (例如，当前没有新快讯)。")
        else:
            print("未能从Apify Actor获取金十数据。请检查Apify控制台中的Actor运行日志。")
        
        print("\n--- Apify 使用说明 ---")
        print(" - 确保您的Apify账户有足够的信用或有效的订阅来运行此Actor。")
        print(" - 您可以在Apify控制台查看Actor的运行日志和详细结果。")
        print(f" - Jinshi News Actor ({JINSHI_NEWS_ACTOR_ID}) 的输入参数 (如有) 请参考其在Apify Store的页面。")

    print("\n所有Apify Jinshi测试结束。") 