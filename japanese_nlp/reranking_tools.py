#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日语重排序模型工具
提供使用日语重排序模型的功能和实用工具
"""

import os
import argparse
import logging
import json
import csv
from typing import List, Dict, Union, Optional, Tuple, Any
import torch
import numpy as np
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def download_nltk_resources():
    """下载NLTK资源"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

class JapaneseReranker:
    """日语重排序模型封装类"""
    
    def __init__(
        self, 
        model_name_or_path: str = "cl-nagoya/ruri-reranker-base",
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        use_fp16: bool = False,
        max_length: int = 512
    ):
        """
        初始化日语重排序模型
        
        Args:
            model_name_or_path: 模型名称或路径
            cache_dir: 缓存目录
            device: 设备（'cpu'或'cuda'）
            use_fp16: 是否使用FP16精度（节省内存）
            max_length: 最大序列长度
        """
        self.model_name = model_name_or_path
        self.max_length = max_length
        
        # 确定设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # 加载模型
        logger.info(f"加载重排序模型: {model_name_or_path}")
        
        # 如果提供了缓存目录，检查是否是本地路径
        if cache_dir is not None and os.path.exists(os.path.join(cache_dir, model_name_or_path.split("/")[-1])):
            model_path = os.path.join(cache_dir, model_name_or_path.split("/")[-1])
            logger.info(f"从本地路径加载模型: {model_path}")
            self.model = CrossEncoder(model_path, device=self.device, max_length=max_length)
        else:
            self.model = CrossEncoder(model_name_or_path, device=self.device, max_length=max_length)
        
        # 设置精度
        if use_fp16 and self.device == "cuda" and hasattr(self.model, "model"):
            self.model.model.half()
        
        logger.info(f"模型加载完成")
    
    def predict(
        self, 
        query: str, 
        passages: List[str], 
        batch_size: int = 32
    ) -> List[Dict[str, Union[int, float, str]]]:
        """
        对查询和文本对进行重排序评分
        
        Args:
            query: 查询文本
            passages: 候选段落列表
            batch_size: 批处理大小
            
        Returns:
            包含排序后的段落信息的列表
        """
        # 构建查询-段落对
        sentence_pairs = [(query, passage) for passage in passages]
        
        # 预测分数
        scores = self.model.predict(sentence_pairs, batch_size=batch_size)
        
        # 构建结果
        results = []
        for i, (passage, score) in enumerate(zip(passages, scores)):
            results.append({
                "id": i,
                "passage": passage,
                "score": float(score)
            })
        
        # 根据分数排序
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        
        return results
    
    def rerank(
        self, 
        query: str, 
        passages: List[str], 
        initial_scores: Optional[List[float]] = None,
        top_k: int = 10,
        batch_size: int = 32,
        alpha: float = 1.0,
        combine_scores: bool = False
    ) -> List[Dict[str, Union[int, float, str]]]:
        """
        对候选段落进行重排序
        
        Args:
            query: 查询文本
            passages: 候选段落列表
            initial_scores: 初始检索分数列表（可选）
            top_k: 返回的最相似段落数量
            batch_size: 批处理大小
            alpha: 合并初始分数和重排序分数的权重（仅在combine_scores=True时使用）
            combine_scores: 是否合并初始分数和重排序分数
            
        Returns:
            包含排序后的段落信息的列表
        """
        # 预测分数并排序
        results = self.predict(query, passages, batch_size=batch_size)
        
        # 合并初始分数和重排序分数
        if combine_scores and initial_scores is not None:
            if len(initial_scores) != len(passages):
                raise ValueError("初始分数列表长度必须与段落列表长度相同")
            
            # 构建id到初始分数的映射
            id_to_initial_score = {i: score for i, score in enumerate(initial_scores)}
            
            # 归一化初始分数
            min_initial_score = min(initial_scores)
            max_initial_score = max(initial_scores)
            range_initial_score = max_initial_score - min_initial_score
            
            if range_initial_score > 0:
                normalized_initial_scores = {
                    i: (score - min_initial_score) / range_initial_score 
                    for i, score in id_to_initial_score.items()
                }
            else:
                normalized_initial_scores = {i: 1.0 for i in range(len(initial_scores))}
            
            # 归一化重排序分数
            min_rerank_score = min(result["score"] for result in results)
            max_rerank_score = max(result["score"] for result in results)
            range_rerank_score = max_rerank_score - min_rerank_score
            
            # 合并分数
            for result in results:
                initial_score = normalized_initial_scores[result["id"]]
                
                if range_rerank_score > 0:
                    rerank_score = (result["score"] - min_rerank_score) / range_rerank_score
                else:
                    rerank_score = 1.0
                
                # 线性插值
                result["original_score"] = result["score"]
                result["score"] = alpha * rerank_score + (1 - alpha) * initial_score
            
            # 重新根据合并后的分数排序
            results = sorted(results, key=lambda x: x["score"], reverse=True)
        
        # 截取top_k个结果
        return results[:top_k]

class BM25Retriever:
    """BM25检索器（作为基线）"""
    
    def __init__(self, tokenize_japanese: bool = False):
        """
        初始化BM25检索器
        
        Args:
            tokenize_japanese: 是否使用日语分词（需要安装mecab-python3）
        """
        self.tokenize_japanese = tokenize_japanese
        self.bm25 = None
        self.corpus = []
        
        # 下载nltk资源（用于英文分词）
        if not self.tokenize_japanese:
            download_nltk_resources()
        
        # 如果使用日语分词，检查MeCab是否可用
        if self.tokenize_japanese:
            try:
                import MeCab
                self.tokenizer = MeCab.Tagger("-Owakati")
            except ImportError:
                logger.warning("MeCab未安装，将回退到使用空格分词。请安装mecab-python3以获得更好的日语分词效果。")
                self.tokenize_japanese = False
    
    def _tokenize(self, text: str) -> List[str]:
        """
        对文本进行分词
        
        Args:
            text: 待分词的文本
            
        Returns:
            分词后的标记列表
        """
        if self.tokenize_japanese:
            # 使用MeCab进行日语分词
            return self.tokenizer.parse(text).strip().split()
        else:
            # 默认使用空格分词，对于日语来说效果不佳
            # 如果是英文，使用nltk分词
            if any(ord(c) > 127 for c in text):  # 包含非ASCII字符，可能是日语
                return text.split()
            else:
                return word_tokenize(text)
    
    def index_documents(self, documents: List[str]):
        """
        对文档进行索引
        
        Args:
            documents: 文档列表
        """
        logger.info(f"对 {len(documents)} 个文档进行BM25索引...")
        self.corpus = documents
        
        # 对文档进行分词
        tokenized_corpus = [self._tokenize(doc) for doc in documents]
        
        # 创建BM25索引
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        logger.info("BM25索引完成")
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Union[int, float, str]]]:
        """
        搜索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回的相关文档数量
            
        Returns:
            包含相关文档信息的列表
        """
        if self.bm25 is None:
            raise ValueError("索引为空，请先调用index_documents()进行索引")
        
        # 对查询进行分词
        tokenized_query = self._tokenize(query)
        
        # 获取BM25分数
        scores = self.bm25.get_scores(tokenized_query)
        
        # 获取排序后的索引
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # 构建结果
        results = []
        for i, idx in enumerate(top_indices):
            results.append({
                "id": int(idx),
                "passage": self.corpus[idx],
                "score": float(scores[idx])
            })
        
        return results

class RerankPipeline:
    """检索-重排序流水线"""
    
    def __init__(
        self, 
        retriever: Union[BM25Retriever, Any],
        reranker: JapaneseReranker,
        top_k_retrieval: int = 100,
        top_k_rerank: int = 10,
        combine_scores: bool = True,
        alpha: float = 0.7
    ):
        """
        初始化检索-重排序流水线
        
        Args:
            retriever: 检索器
            reranker: 重排序器
            top_k_retrieval: 检索阶段返回的文档数量
            top_k_rerank: 重排序阶段返回的文档数量
            combine_scores: 是否合并检索分数和重排序分数
            alpha: 合并分数的权重（仅在combine_scores=True时使用）
        """
        self.retriever = retriever
        self.reranker = reranker
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank
        self.combine_scores = combine_scores
        self.alpha = alpha
    
    def search(self, query: str) -> List[Dict[str, Union[int, float, str]]]:
        """
        执行检索-重排序搜索
        
        Args:
            query: 查询文本
            
        Returns:
            包含排序后的文档信息的列表
        """
        # 第一阶段：检索
        logger.info(f"执行检索: {query}")
        retrieval_results = self.retriever.search(query, top_k=self.top_k_retrieval)
        
        # 提取段落和分数
        passages = [result["passage"] for result in retrieval_results]
        initial_scores = [result["score"] for result in retrieval_results]
        
        # 第二阶段：重排序
        logger.info("执行重排序...")
        rerank_results = self.reranker.rerank(
            query=query,
            passages=passages,
            initial_scores=initial_scores,
            top_k=self.top_k_rerank,
            combine_scores=self.combine_scores,
            alpha=self.alpha
        )
        
        # 映射回原始文档ID
        for result in rerank_results:
            result["original_id"] = retrieval_results[result["id"]]["id"]
        
        return rerank_results

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="日语重排序模型工具")
    
    # 子命令
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # 预测命令
    predict_parser = subparsers.add_parser("predict", help="预测查询-段落对的得分")
    predict_parser.add_argument("--query", type=str, required=True, help="查询文本")
    predict_parser.add_argument("--passages", type=str, required=True, help="候选段落文件路径（每行一个段落）")
    predict_parser.add_argument("--model", type=str, default="cl-nagoya/ruri-reranker-base", help="重排序模型名称或路径")
    predict_parser.add_argument("--output", type=str, help="输出文件路径（JSON格式）")
    predict_parser.add_argument("--batch-size", type=int, default=32, help="批处理大小")
    
    # 重排序命令
    rerank_parser = subparsers.add_parser("rerank", help="重排序候选段落")
    rerank_parser.add_argument("--query", type=str, required=True, help="查询文本")
    rerank_parser.add_argument("--passages", type=str, required=True, help="候选段落文件路径（每行一个段落）")
    rerank_parser.add_argument("--scores", type=str, help="初始分数文件路径（每行一个分数，与段落一一对应）")
    rerank_parser.add_argument("--model", type=str, default="cl-nagoya/ruri-reranker-base", help="重排序模型名称或路径")
    rerank_parser.add_argument("--top-k", type=int, default=10, help="返回的最相似段落数量")
    rerank_parser.add_argument("--batch-size", type=int, default=32, help="批处理大小")
    rerank_parser.add_argument("--output", type=str, help="输出文件路径（JSON格式）")
    rerank_parser.add_argument("--alpha", type=float, default=0.7, help="合并分数的权重")
    rerank_parser.add_argument("--combine-scores", action="store_true", help="是否合并初始分数和重排序分数")
    
    # 流水线命令
    pipeline_parser = subparsers.add_parser("pipeline", help="执行检索-重排序流水线")
    pipeline_parser.add_argument("--query", type=str, required=True, help="查询文本")
    pipeline_parser.add_argument("--corpus", type=str, required=True, help="语料库文件路径（每行一个文档）")
    pipeline_parser.add_argument("--model", type=str, default="cl-nagoya/ruri-reranker-base", help="重排序模型名称或路径")
    pipeline_parser.add_argument("--retrieval-top-k", type=int, default=100, help="检索阶段返回的文档数量")
    pipeline_parser.add_argument("--rerank-top-k", type=int, default=10, help="重排序阶段返回的文档数量")
    pipeline_parser.add_argument("--batch-size", type=int, default=32, help="批处理大小")
    pipeline_parser.add_argument("--output", type=str, help="输出文件路径（JSON格式）")
    pipeline_parser.add_argument("--tokenize-japanese", action="store_true", help="是否使用日语分词（需要安装mecab-python3）")
    pipeline_parser.add_argument("--alpha", type=float, default=0.7, help="合并分数的权重")
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    if args.command == "predict":
        # 加载段落
        with open(args.passages, "r", encoding="utf-8") as f:
            passages = [line.strip() for line in f]
        
        # 创建重排序模型
        reranker = JapaneseReranker(model_name_or_path=args.model)
        
        # 预测分数
        results = reranker.predict(args.query, passages, batch_size=args.batch_size)
        
        # 输出结果
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"结果已保存到: {args.output}")
        else:
            logger.info(f"查询: {args.query}")
            logger.info(f"找到 {len(results)} 个结果:")
            for i, result in enumerate(results[:10], 1):  # 只显示前10个结果
                logger.info(f"{i}. 分数: {result['score']:.4f}")
                logger.info(f"   文本: {result['passage'][:100]}...")
    
    elif args.command == "rerank":
        # 加载段落
        with open(args.passages, "r", encoding="utf-8") as f:
            passages = [line.strip() for line in f]
        
        # 加载初始分数（如果提供）
        initial_scores = None
        if args.scores and args.combine_scores:
            with open(args.scores, "r", encoding="utf-8") as f:
                initial_scores = [float(line.strip()) for line in f]
            
            if len(initial_scores) != len(passages):
                logger.error(f"初始分数数量 ({len(initial_scores)}) 与段落数量 ({len(passages)}) 不匹配")
                return 1
        
        # 创建重排序模型
        reranker = JapaneseReranker(model_name_or_path=args.model)
        
        # 重排序
        results = reranker.rerank(
            args.query, 
            passages, 
            initial_scores=initial_scores,
            top_k=args.top_k,
            batch_size=args.batch_size,
            combine_scores=args.combine_scores,
            alpha=args.alpha
        )
        
        # 输出结果
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"结果已保存到: {args.output}")
        else:
            logger.info(f"查询: {args.query}")
            logger.info(f"找到 {len(results)} 个结果:")
            for i, result in enumerate(results, 1):
                logger.info(f"{i}. 分数: {result['score']:.4f}")
                logger.info(f"   文本: {result['passage'][:100]}...")
    
    elif args.command == "pipeline":
        # 加载语料库
        with open(args.corpus, "r", encoding="utf-8") as f:
            corpus = [line.strip() for line in f]
        
        # 创建检索器
        retriever = BM25Retriever(tokenize_japanese=args.tokenize_japanese)
        retriever.index_documents(corpus)
        
        # 创建重排序模型
        reranker = JapaneseReranker(model_name_or_path=args.model)
        
        # 创建流水线
        pipeline = RerankPipeline(
            retriever=retriever,
            reranker=reranker,
            top_k_retrieval=args.retrieval_top_k,
            top_k_rerank=args.rerank_top_k,
            combine_scores=True,
            alpha=args.alpha
        )
        
        # 执行搜索
        results = pipeline.search(args.query)
        
        # 输出结果
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"结果已保存到: {args.output}")
        else:
            logger.info(f"查询: {args.query}")
            logger.info(f"找到 {len(results)} 个结果:")
            for i, result in enumerate(results, 1):
                logger.info(f"{i}. 分数: {result['score']:.4f}")
                logger.info(f"   原始分数: {result.get('original_score', 'N/A')}")
                logger.info(f"   文本: {result['passage'][:100]}...")
    
    else:
        logger.error("请指定子命令: predict, rerank, pipeline")
        return 1

if __name__ == "__main__":
    main() 