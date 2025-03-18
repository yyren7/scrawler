# 人物信息爬取与分析系统

这是一个使用LangChain和Tavily API的人物信息爬取与分析工具。该工具可以根据用户提供的公司名称和人物的部分名字，从互联网上搜集相关信息，并根据用户指定的分析方向生成结构化的分析报告。本系统使用DeepSeek API进行内容生成，使用BAAI/bge-m3嵌入模型进行向量检索。

## 功能特点

- 基于公司名称和部分姓名进行智能搜索
- 支持自定义分析方向（如职业经历、教育背景、技术专长等）
- 使用向量数据库进行高效信息检索
- 利用DeepSeek LLM进行智能分析和总结
- 采用BAAI/bge-m3嵌入模型提高检索准确性

## 安装方法

1. 克隆此仓库:
```
git clone https://github.com/yourusername/scrawler.git
cd scrawler
```

2. 安装依赖:
```
pip install -r requirements.txt
```

3. 设置必要的环境变量:
```
# Linux/Mac
export TAVILY_API_KEY="your-tavily-api-key"
export DEEPSEEK_API_KEY="your-deepseek-api-key"
export DEEPSEEK_API_BASE="https://api.deepseek.com"  # DeepSeek API的基础URL

# Windows (PowerShell)
$env:TAVILY_API_KEY="your-tavily-api-key"
$env:DEEPSEEK_API_KEY="your-deepseek-api-key"
$env:DEEPSEEK_API_BASE="https://api.deepseek.com"  # DeepSeek API的基础URL
```

## 使用方法

运行主程序:
```
python interviewer_info_scrawler/scralwer.py
```

按照提示输入:
1. 目标人物的公司
2. 目标人物的部分名字
3. 分析方向(如:职业经历、教育背景、技术专长等)

系统将自动搜索相关信息，并生成分析报告。

## 依赖项

- langchain==0.3.20
- langchain-community==0.3.19
- langchain-core==0.3.42
- langchain-huggingface>=0.0.2  # 用于嵌入模型
- openai>=1.0.0  # 用于与DeepSeek API通信
- tavily-python==0.5.1
- faiss-cpu==1.10.0
- streamlit==1.43.1
- sentence-transformers>=2.2.0  # 用于BGE-M3嵌入模型