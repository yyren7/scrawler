#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日语NLP模型安装脚本
安装最新和最佳的日语提示扩展、嵌入和重排序模型
"""

import os
import argparse
import logging
import subprocess
import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 定义模型列表
EMBEDDING_MODELS = {
    # 最新的日语嵌入模型
    "large": "cl-nagoya/ruri-pt-large",  # 最新的Ruri日语嵌入模型（大型）
    "base": "cl-nagoya/ruri-base",  # Ruri日语嵌入模型（基础）
    "small": "cl-nagoya/ruri-pt-small",  # Ruri日语嵌入模型（小型）
    "multilingual": "Alibaba-NLP/gte-Qwen2-7B-instruct",  # 多语言嵌入模型，支持日语
    "legacy": "oshizo/sbert-jsnli-luke-japanese-base-lite"  # 较早的日语SBERT模型
}

RERANKING_MODELS = {
    # 日语重排序模型
    "large": "cl-nagoya/ruri-reranker-large",  # 最新的Ruri日语重排序模型（大型）
    "base": "cl-nagoya/ruri-reranker-base",  # Ruri日语重排序模型（基础）
    "small": "cl-nagoya/ruri-reranker-small",  # Ruri日语重排序模型（小型）
    "cross-encoder": "hotchpotch/japanese-reranker-cross-encoder-base-v1"  # 日语交叉编码器
}

LLM_MODELS = {
    # 支持日语提示扩展的LLM模型
    "qwen": "Qwen/Qwen2-72B-Instruct",  # 支持日语的Qwen2模型
    "llama": "tokyotech-llm/Llama-3-Swallow-8B-v0.1",  # 支持日语的Llama3衍生模型
    "open-source": "SakanaAI/EvoLLM-JP-v1-7B",  # 开源日语模型
    "small": "rinna/nekomata-7b-instruction"  # 较小的日语指令模型
}

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="安装日语NLP模型")
    
    parser.add_argument("--embedding", type=str, choices=list(EMBEDDING_MODELS.keys()) + ["all"], 
                        default="base", help="要安装的嵌入模型")
    
    parser.add_argument("--reranking", type=str, choices=list(RERANKING_MODELS.keys()) + ["all"], 
                        default="base", help="要安装的重排序模型")
    
    parser.add_argument("--llm", type=str, choices=list(LLM_MODELS.keys()) + ["all", "none"], 
                        default="none", help="要安装的LLM模型（可选）")
    
    parser.add_argument("--skip-embedding", action="store_true", 
                        help="跳过安装嵌入模型")
    
    parser.add_argument("--skip-reranking", action="store_true", 
                        help="跳过安装重排序模型")
    
    parser.add_argument("--cache-dir", type=str, default="./models",
                        help="模型缓存目录")
    
    parser.add_argument("--force-download", action="store_true",
                        help="强制重新下载模型")
    
    return parser.parse_args()

def install_requirements():
    """安装依赖"""
    logger.info("安装依赖...")
    try:
        subprocess.check_call(
            ["pip", "install", "-r", "requirements.txt"]
        )
        logger.info("依赖安装完成")
    except subprocess.CalledProcessError as e:
        logger.error(f"安装依赖失败: {e}")
        raise

def download_model(model_name, cache_dir, force_download=False):
    """下载模型"""
    logger.info(f"下载模型: {model_name}")
    try:
        model_path = os.path.join(cache_dir, model_name.split("/")[-1])
        
        # 如果模型已存在且不强制下载，则跳过
        if os.path.exists(model_path) and not force_download:
            logger.info(f"模型 {model_name} 已存在，跳过下载")
            return model_path
        
        # 从Hugging Face下载模型
        snapshot_download(
            repo_id=model_name,
            local_dir=model_path,
            local_dir_use_symlinks=False
        )
        
        logger.info(f"模型 {model_name} 下载完成")
        return model_path
    except Exception as e:
        logger.error(f"下载模型 {model_name} 失败: {e}")
        raise

def install_embedding_models(model_key, cache_dir, force_download=False):
    """安装嵌入模型"""
    os.makedirs(cache_dir, exist_ok=True)
    
    models_to_install = []
    if model_key == "all":
        models_to_install = list(EMBEDDING_MODELS.values())
    else:
        models_to_install = [EMBEDDING_MODELS[model_key]]
    
    for model_name in models_to_install:
        download_model(model_name, cache_dir, force_download)
        
        # 尝试加载模型以验证下载是否成功
        try:
            from sentence_transformers import SentenceTransformer
            model_path = os.path.join(cache_dir, model_name.split("/")[-1])
            model = SentenceTransformer(model_path)
            embeddings = model.encode("このモデルのテスト文です。", convert_to_tensor=True)
            logger.info(f"嵌入模型 {model_name} 测试成功，输出维度: {embeddings.shape}")
        except Exception as e:
            logger.error(f"加载嵌入模型 {model_name} 失败: {e}")
            raise

def install_reranking_models(model_key, cache_dir, force_download=False):
    """安装重排序模型"""
    os.makedirs(cache_dir, exist_ok=True)
    
    models_to_install = []
    if model_key == "all":
        models_to_install = list(RERANKING_MODELS.values())
    else:
        models_to_install = [RERANKING_MODELS[model_key]]
    
    for model_name in models_to_install:
        download_model(model_name, cache_dir, force_download)
        
        # 尝试加载模型以验证下载是否成功
        try:
            from sentence_transformers import CrossEncoder
            model_path = os.path.join(cache_dir, model_name.split("/")[-1])
            model = CrossEncoder(model_path)
            scores = model.predict([("これは質問です。", "これは回答です。")])
            logger.info(f"重排序模型 {model_name} 测试成功，输出得分: {scores}")
        except Exception as e:
            logger.error(f"加载重排序模型 {model_name} 失败: {e}")
            raise

def install_llm_models(model_key, cache_dir, force_download=False):
    """安装LLM模型（可选）"""
    if model_key == "none":
        logger.info("跳过安装LLM模型")
        return
    
    os.makedirs(cache_dir, exist_ok=True)
    
    models_to_install = []
    if model_key == "all":
        models_to_install = list(LLM_MODELS.values())
    else:
        models_to_install = [LLM_MODELS[model_key]]
    
    for model_name in models_to_install:
        download_model(model_name, cache_dir, force_download)
        
        # 注意：我们不在这里加载LLM模型进行测试，因为它们可能很大

def main():
    args = parse_args()
    
    # 安装依赖
    install_requirements()
    
    # 创建缓存目录
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # 安装嵌入模型
    if not args.skip_embedding:
        logger.info(f"开始安装嵌入模型: {args.embedding}")
        install_embedding_models(args.embedding, args.cache_dir, args.force_download)
    
    # 安装重排序模型
    if not args.skip_reranking:
        logger.info(f"开始安装重排序模型: {args.reranking}")
        install_reranking_models(args.reranking, args.cache_dir, args.force_download)
    
    # 安装LLM模型（可选）
    if args.llm != "none":
        logger.info(f"开始安装LLM模型: {args.llm}")
        install_llm_models(args.llm, args.cache_dir, args.force_download)
    
    logger.info("所有模型安装完成！")

if __name__ == "__main__":
    main() 