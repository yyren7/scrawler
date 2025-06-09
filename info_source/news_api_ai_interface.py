# Interface for accessing NewsAPI.ai (Event Registry) data.
# analyze_tool.md lists NewsAPI.ai for global news with metadata (entities, topics, sentiment).
# The service appears to use the Event Registry backend and Python SDK.

import os
import json
from datetime import datetime, timedelta
from eventregistry import ( # type: ignore
    EventRegistry, QueryArticlesIter, QueryItems, ReturnInfo, ArticleInfoFlags,
    SourceInfoFlags, ConceptInfoFlags, CategoryInfoFlags, LocationInfoFlags
)
from dotenv import load_dotenv # Added for .env loading

load_dotenv() # Load environment variables from .env file

# API key is now primarily loaded from .env, with a fallback to the placeholder.
NEWSAPI_AI_KEY = os.environ.get("NEWSAPI_AI_KEY") or "YOUR_API_KEY_HERE"

# Global EventRegistry client instance
er_client_instance = None

def get_newsapi_ai_client():
    global er_client_instance
    if NEWSAPI_AI_KEY == "YOUR_API_KEY_HERE" or NEWSAPI_AI_KEY is None:
        print("错误：NEWSAPI_AI_KEY 未配置。请在 .env 文件中或直接在脚本中设置API密钥。")
        print("请访问 https://newsapi.ai/ 或 https://eventregistry.org/ 获取API密钥。")
        return None
    if er_client_instance is None:
        try:
            # allowUseOfArchive=False by default for free keys, True for paid.
            # We'll let the API decide based on the key's permissions by default.
            er_client_instance = EventRegistry(apiKey=NEWSAPI_AI_KEY, allowUseOfArchive=True) 
        except Exception as e:
            print(f"初始化NewsAPI.ai (EventRegistry)客户端时发生错误: {e}")
            er_client_instance = None
            return None
    return er_client_instance

def search_articles(keywords=None, concept_uris=None, category_uris=None, source_uris=None, 
                    source_location_uris=None, author_uris=None,
                    lang='eng', date_start=None, date_end=None, 
                    min_sentiment=None, max_sentiment=None,
                    sort_by="date", sort_by_asc=False, max_items=10):
    """
    Searches for news articles using NewsAPI.ai (Event Registry).

    Args:
        keywords (str or QueryItems, optional): Keywords to search for.
        concept_uris (list or str, optional): List of concept URIs or a single URI.
        category_uris (list or str, optional): List of category URIs or a single URI.
        source_uris (list or str, optional): List of source URIs or a single URI.
        source_location_uris (list or str, optional): List of source location URIs or a single URI.
        author_uris (list or str, optional): List of author URIs or a single URI.
        lang (str or list, optional): Language code(s) (e.g., 'eng', ['eng', 'deu']). Defaults to 'eng'.
        date_start (str or datetime, optional): Start date (YYYY-MM-DD or datetime object).
        date_end (str or datetime, optional): End date (YYYY-MM-DD or datetime object).
        min_sentiment (float, optional): Minimum sentiment score (-1 to 1).
        max_sentiment (float, optional): Maximum sentiment score (-1 to 1).
        sort_by (str, optional): Sorting criteria (e.g., "date", "rel", "socialScore"). Defaults to "date".
        sort_by_asc (bool, optional): Sort ascending. Defaults to False.
        max_items (int, optional): Maximum number of articles to return. Defaults to 10.

    Returns:
        list: A list of article dictionaries, or None if an error occurs.
    """
    er = get_newsapi_ai_client()
    if not er:
        return None

    # Construct the query
    # QueryArticlesIter is used for iterating over results, especially when there are many.
    # For a limited number of items, execQuery directly on a QueryArticles object can also be used.
    q = QueryArticlesIter(
        keywords=keywords,
        conceptUri=concept_uris,
        categoryUri=category_uris,
        sourceUri=source_uris,
        sourceLocationUri=source_location_uris,
        authorUri=author_uris,
        dateStart=date_start,
        dateEnd=date_end,
        lang=lang,
        minSentiment=min_sentiment,
        maxSentiment=max_sentiment
    )

    # Define what information to return for each article
    # This is crucial for getting the rich metadata NewsAPI.ai offers
    ri = ReturnInfo(
        articleInfo=ArticleInfoFlags(
            bodyLen=-1, # Return full body
            concepts=True, 
            categories=True, 
            links=True, 
            videos=True, 
            image=True, 
            socialScore=True, 
            sentiment=True, # Request article sentiment
            emotions=True,  # Request article emotions
            duplicateList=True,
            originalArticle=True,
            authors=True,
            source=SourceInfoFlags(title=True, domain=True, location=True, ranking=True),
            # Basic concept info within article data
            # conceptInfo=ConceptInfoFlags(type="person", lang="eng", synonyms=True, image=True, description=True),
            # categoryInfo=CategoryInfoFlags(parentUri=True, childrenUris=True),
            # locationInfo=LocationInfoFlags(countryIso2=True, placeFeatureCode=True),
        )
    )

    results = []
    try:
        # execQuery returns a generator
        for article in q.execQuery(er, sortBy=sort_by, sortByAsc=sort_by_asc, returnInfo=ri, maxItems=max_items):
            results.append(article)
        
        # The SDK handles many API errors by raising exceptions.
        # If results is empty, it means no articles matched, or maxItems was 0.
        if not results and max_items > 0:
            print(f"NewsAPI.ai: 未找到符合查询条件的新闻。查询参数: keywords={keywords}, concepts={concept_uris}, date_start={date_start}, etc.")
        
        return results

    except Exception as e:
        print(f"调用NewsAPI.ai (search_articles) 时发生错误: {e}")
        # Specific error handling for EventRegistry might be needed if its exceptions are distinct
        # For example, check for API key errors or rate limits if the exception provides such details.
        return None

# --- 基础测试 ---
if __name__ == "__main__":
    print("--- 测试 NewsAPI.ai (Event Registry) API ---")

    client_init_success = bool(get_newsapi_ai_client())

    if not client_init_success:
        print("由于API密钥未配置或客户端初始化失败，跳过NewsAPI.ai的实时测试。")
    else:
        print(f"使用的API密钥: {'已配置' if NEWSAPI_AI_KEY != 'YOUR_API_KEY_HERE' else '未配置或为占位符'}")
        print("注意: NewsAPI.ai/Event Registry的免费/试用套餐对 'tokens' 和历史数据访问有严格限制。")

        # Test 1: Recent news about a major company
        print("\n测试1: 获取关于 'Microsoft' 的最新2条新闻 (过去7天)")
        today = datetime.now()
        seven_days_ago = today - timedelta(days=7)
        msft_news = search_articles(
            keywords="Microsoft", 
            lang="eng", 
            date_start=seven_days_ago.strftime("%Y-%m-%d"), 
            date_end=today.strftime("%Y-%m-%d"), 
            max_items=2
        )

        if msft_news is not None:
            if msft_news:
                print(f"获取到 {len(msft_news)} 条关于 Microsoft 的新闻:")
                for article in msft_news:
                    print(f"  标题: {article.get('title')}")
                    print(f"  来源: {article.get('source', {}).get('title')} ({article.get('source', {}).get('domain')})")
                    print(f"  发布日期: {article.get('date')}")
                    print(f"  URL: {article.get('url')}")
                    print(f"  情感评分: {article.get('sentiment')}")
                    if article.get('concepts') and len(article.get('concepts', [])) > 0:
                        print(f"  主要概念: {[(concept.get('label',{}).get('eng','N/A'), concept.get('type')) for concept in article['concepts'][:3]]}") # Show top 3 concepts
                    print("----")
            else:
                print("未找到 Microsoft 的新闻。")
        else:
            print("获取 Microsoft 新闻失败。")

        # Test 2: News by a specific concept URI (if known, otherwise keyword is easier for general use)
        # To get concept URIs, one might first use EventRegistry().getConceptUri("Keyword")
        er_client = get_newsapi_ai_client()
        if er_client:
            try:
                oil_concept_uri = er_client.getConceptUri("crude oil")
                if oil_concept_uri:
                    print(f"\n测试2: 获取关于概念 'crude oil' (URI: {oil_concept_uri}) 的最新1条新闻")
                    oil_news = search_articles(concept_uris=[oil_concept_uri], max_items=1)
                    if oil_news:
                        for article in oil_news:
                            print(f"  标题: {article.get('title')}")
                            print(f"  来源: {article.get('source', {}).get('title')}")
                            print("----")
                    else:
                        print("未找到 'crude oil' 相关概念的新闻或获取失败。")
                else:
                    print("未能获取 'crude oil' 的概念URI，跳过测试2。")
            except Exception as e:
                 print(f"获取概念URI或搜索时出错 (测试2): {e}")
        else:
            print("NewsAPI.ai客户端未初始化，跳过测试2。")
            
        # Test 3: News with sentiment filter (example)
        print("\n测试3: 获取关于 'NVIDIA' 的正面新闻 (sentiment > 0.5) (最多1条)")
        nvidia_positive_news = search_articles(
            keywords="NVIDIA",
            lang="eng",
            date_start=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"), # Broader date range for sentiment
            date_end=datetime.now().strftime("%Y-%m-%d"),
            min_sentiment=0.5, 
            max_items=1, 
            sort_by="rel" # Sort by relevance for keyword search
        )
        if nvidia_positive_news:
            for article in nvidia_positive_news:
                print(f"  标题: {article.get('title')}")
                print(f"  情感评分: {article.get('sentiment')}")
                print("----")
        else:
            print("未找到 NVIDIA 的高正面情感新闻或获取失败。")

    print("\n所有 NewsAPI.ai (Event Registry) 测试结束。")
    print("提醒: 此API的免费/试用额度基于'tokens'消耗，不同查询的token成本不同。历史数据和复杂查询消耗更多。") 