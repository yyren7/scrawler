import os
from typing import List, Dict, Any, Optional, Mapping
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import openai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

def get_tavily_api_key():
    """从环境变量获取Tavily API密钥"""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("请设置TAVILY_API_KEY环境变量")
    return api_key

def get_deepseek_api_key():
    """从环境变量获取DeepSeek API密钥"""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("请设置DEEPSEEK_API_KEY环境变量")
    return api_key

def get_deepseek_api_base():
    """从环境变量获取DeepSeek API基础URL"""
    api_base = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com")
    return api_base

# 创建DeepSeek客户端
def create_deepseek_client():
    """创建DeepSeek API客户端"""
    client = openai.OpenAI(
        api_key=get_deepseek_api_key(),
        base_url=get_deepseek_api_base() + "/v1"
    )
    return client

# 创建DeepSeek Chat模型类
class DeepSeekChatModel(BaseChatModel):
    """DeepSeek聊天模型包装器"""
    
    model_name: str = "deepseek-chat"
    client: Any = None
    
    def __init__(self, model_name: str = "deepseek-chat"):
        super().__init__()
        self.model_name = model_name
        self.client = create_deepseek_client()
    
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # 将LangChain消息类型映射到DeepSeek API支持的角色类型
        message_dicts = []
        for m in messages:
            # 将LangChain的消息类型映射到DeepSeek API支持的角色
            role = m.type
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"
                
            message_dicts.append({"role": role, "content": m.content})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=message_dicts,
                stop=stop,
                **kwargs
            )
            
            message = AIMessage(content=response.choices[0].message.content)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
        except Exception as e:
            print(f"DeepSeek API调用错误: {e}")
            # 尝试使用更简单的消息格式再次调用
            try:
                simple_messages = [{"role": "user", "content": "请简单回答：Hello"}]
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=simple_messages
                )
                print("使用简化消息成功，原始消息格式可能有问题")
                message = AIMessage(content="由于API调用问题，无法生成回答。请检查DeepSeek API的设置和格式要求。")
                generation = ChatGeneration(message=message)
                return ChatResult(generations=[generation])
            except Exception as e2:
                print(f"简化消息调用也失败: {e2}")
                # 返回一个错误消息
                message = AIMessage(content="API调用失败。请检查API密钥和连接。")
                generation = ChatGeneration(message=message)
                return ChatResult(generations=[generation])
    
    @property
    def _llm_type(self) -> str:
        return "deepseek-chat"
        
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name}
        
    def get_num_tokens(self, text: str) -> int:
        """估算文本的token数量"""
        # 粗略估计，每4个字符约为1个token
        return len(text) // 4

def search_person_info(company: str, partial_name: str, search_direction: str) -> List[Dict[Any, Any]]:
    """
    使用Tavily搜索有关指定人物的信息
    
    参数:
        company: 公司名称
        partial_name: 人物的部分名字
        search_direction: 搜索方向/主题
        
    返回:
        搜索结果列表
    """
    try:
        search_query = f"{partial_name} {company} {search_direction}"
        print(f"正在搜索: {search_query}")
        
        tavily_tool = TavilySearchResults(max_results=10)
        search_results = tavily_tool.invoke({"query": search_query})
        
        if not search_results:
            print("没有找到搜索结果")
            return []
            
        print(f"找到 {len(search_results)} 条结果")
        # 打印搜索结果的简介到控制台
        print("\n搜索结果简介:")
        for i, result in enumerate(search_results, 1):
            title = result.get('title', '无标题')
            url = result.get('url', '无链接')
            # 获取内容的前100个字符作为简介
            content_preview = result.get('content', '')[:100] + ('...' if len(result.get('content', '')) > 100 else '')
            print(f"{i}. 标题: {title}")
            print(f"   链接: {url}")
            print(f"   简介: {content_preview}")
            print("-" * 50)

        return search_results
    except Exception as e:
        print(f"搜索过程中出错: {e}")
        return []

def create_vector_db(search_results: List[Dict[Any, Any]]):
    """
    从搜索结果创建向量数据库
    
    参数:
        search_results: 搜索结果列表
        
    返回:
        FAISS向量存储
    """
    # 从搜索结果中提取文本
    texts = []
    for result in search_results:
        texts.append(f"标题: {result.get('title', '')}\n内容: {result.get('content', '')}\n链接: {result.get('url', '')}")
    
    # 切分文本
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.create_documents(texts)
    
    # 创建向量存储，使用BGE-M3嵌入模型
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def create_llm_model():
    """
    创建DeepSeek LLM模型
    
    返回:
        DeepSeek LLM模型
    """
    return DeepSeekChatModel(model_name="deepseek-chat")

def analyze_person(company: str, partial_name: str, search_direction: str) -> str:
    """
    分析指定人物的信息
    
    参数:
        company: 公司名称
        partial_name: 人物的部分名字
        search_direction: 搜索方向/主题
        
    返回:
        分析结果文本
    """
    # 1. 搜索信息
    search_results = search_person_info(company, partial_name, search_direction)
    if not search_results:
        return "无法找到有关此人的信息。"
    
    # 2. 创建向量数据库
    vector_store = create_vector_db(search_results)
    
    # 3. 创建检索链
    retriever = vector_store.as_retriever()
    
    # 4. 设置提示模板
    prompt = ChatPromptTemplate.from_template("""
    你是一位专业的人物分析师。基于以下关于一个人的信息，请提供关于他们的详细分析。
    分析重点: {search_direction}
    
    相关信息:
    {context}
    
    请以结构化方式组织你的回答，包括:
    1. 个人基本情况概述
    2. 与分析重点相关的详细信息
    3. 可能的见解和结论
    
    分析:
    """)
    
    # 5. 创建文档链和检索链，使用DeepSeek模型
    llm = create_llm_model()
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # 6. 执行分析
    response = retrieval_chain.invoke({
        "input": search_direction,
        "search_direction": search_direction
    })
    
    return response["answer"]

def main():
    print("====== 人物信息爬取与分析系统 ======")
    
    # 检查API密钥
    try:
        get_tavily_api_key()
        get_deepseek_api_key()
    except ValueError as e:
        print(f"错误: {e}")
        print("请设置必要的API密钥后再运行程序。")
        return
    
    # 获取用户输入
    company = input("请输入目标人物的公司: ")
    partial_name = input("请输入目标人物的部分名字: ")
    search_direction = input("请输入分析方向(如:职业经历、教育背景、技术专长等): ")
    
    print("\n开始分析，请稍候...\n")
    start_time = time.time()
    
    # 执行分析
    result = analyze_person(company, partial_name, search_direction)
    
    print("\n===== 分析结果 =====")
    print(result)
    print(f"\n分析完成，耗时: {time.time() - start_time:.2f}秒")

if __name__ == "__main__":
    main()


