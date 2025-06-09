#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日语嵌入模型工具
提供使用日语嵌入模型的功能和实用工具
"""

import os
import argparse
import logging
import numpy as np
import torch
from typing import List, Dict, Union, Optional, Tuple
from sentence_transformers import SentenceTransformer, util
import faiss

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class JapaneseEmbedder:
    """日语嵌入模型封装类"""
    
    def __init__(
        self, 
        model_name_or_path: str = "cl-nagoya/ruri-base",
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        use_fp16: bool = False
    ):
        """
        初始化日语嵌入模型
        
        Args:
            model_name_or_path: 模型名称或路径
            cache_dir: 缓存目录
            device: 设备（'cpu'或'cuda'）
            use_fp16: 是否使用FP16精度（节省内存）
        """
        self.model_name = model_name_or_path
        
        # 确定设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # 加载模型
        logger.info(f"加载嵌入模型: {model_name_or_path}")
        
        # 如果提供了缓存目录，检查是否是本地路径
        if cache_dir is not None and os.path.exists(os.path.join(cache_dir, model_name_or_path.split("/")[-1])):
            model_path = os.path.join(cache_dir, model_name_or_path.split("/")[-1])
            logger.info(f"从本地路径加载模型: {model_path}")
            self.model = SentenceTransformer(model_path, device=self.device)
        else:
            self.model = SentenceTransformer(model_name_or_path, device=self.device)
        
        # 设置精度
        if use_fp16 and self.device == "cuda":
            self.model.half()
            
        # 自动确定输出维度
        self.embedding_dim = self._get_embedding_dimension()
        logger.info(f"模型加载完成，输出维度: {self.embedding_dim}")
    
    def _get_embedding_dimension(self) -> int:
        """获取嵌入向量的维度"""
        test_embedding = self.embed("テスト文", convert_to_numpy=True)
        return test_embedding.shape[0]
    
    def embed(
        self, 
        text: Union[str, List[str]], 
        batch_size: int = 32,
        normalize: bool = True,
        convert_to_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor, List[Union[np.ndarray, torch.Tensor]]]:
        """
        对文本进行嵌入
        
        Args:
            text: 待嵌入的文本或文本列表
            batch_size: 批处理大小
            normalize: 是否对输出向量进行归一化
            convert_to_numpy: 是否将输出转换为NumPy数组
            
        Returns:
            嵌入向量
        """
        return self.model.encode(
            text, 
            batch_size=batch_size,
            convert_to_tensor=not convert_to_numpy,
            normalize_embeddings=normalize
        )
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            相似度分数（0-1）
        """
        emb1 = self.embed(text1, convert_to_numpy=False)
        emb2 = self.embed(text2, convert_to_numpy=False)
        
        return util.cos_sim(emb1, emb2).item()
    
    def semantic_search(
        self, 
        query: str, 
        corpus: List[str], 
        top_k: int = 5
    ) -> List[Dict[str, Union[int, float, str]]]:
        """
        在语料库中进行语义搜索
        
        Args:
            query: 查询文本
            corpus: 语料库文本列表
            top_k: 返回的最相似文本数量
            
        Returns:
            包含相似文本信息的列表
        """
        query_embedding = self.embed(query, convert_to_numpy=False)
        corpus_embeddings = self.embed(corpus, convert_to_numpy=False)
        
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
        hits = hits[0]  # 仅考虑第一个查询
        
        results = []
        for hit in hits:
            results.append({
                "corpus_id": hit["corpus_id"],
                "score": hit["score"],
                "text": corpus[hit["corpus_id"]]
            })
        
        return results

class FaissIndexer:
    """使用FAISS的高效索引和检索"""
    
    def __init__(
        self, 
        embedder: JapaneseEmbedder,
        index_type: str = "Flat"
    ):
        """
        初始化FAISS索引器
        
        Args:
            embedder: 嵌入模型
            index_type: 索引类型（'Flat'或'IVF'或'HNSW'）
        """
        self.embedder = embedder
        self.index_type = index_type
        self.dim = embedder.embedding_dim
        self.index = None
        self.corpus = []
        
        # 创建索引
        self._create_index()
    
    def _create_index(self):
        """创建FAISS索引"""
        if self.index_type == "Flat":
            self.index = faiss.IndexFlatIP(self.dim)  # 内积（余弦相似度，假设向量已归一化）
        elif self.index_type == "IVF":
            # 创建IVF索引，通常需要更多数据才能有效训练
            quantizer = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, 100, faiss.METRIC_INNER_PRODUCT)
            self.index.nprobe = 10  # 默认搜索10个单元
        elif self.index_type == "HNSW":
            # 创建HNSW索引，适合大规模近似最近邻搜索
            self.index = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 128
        else:
            raise ValueError(f"不支持的索引类型: {self.index_type}")
    
    def index_documents(self, documents: List[str], batch_size: int = 256):
        """
        对文档进行索引
        
        Args:
            documents: 文档列表
            batch_size: 批处理大小
        """
        logger.info(f"对 {len(documents)} 个文档进行索引...")
        self.corpus = documents
        
        # 计算嵌入向量
        embeddings = self.embedder.embed(documents, batch_size=batch_size, normalize=True)
        
        # 如果是IVF索引，则需要先训练
        if self.index_type == "IVF" and not self.index.is_trained:
            if len(embeddings) < 100:
                logger.warning("IVF索引通常需要大量数据才能有效训练。对于小型语料库，建议使用Flat索引。")
            self.index.train(embeddings)
        
        # 将向量添加到索引中
        self.index.add(embeddings)
        logger.info(f"索引完成，共 {self.index.ntotal} 个向量")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Union[int, float, str]]]:
        """
        搜索最相似的文档
        
        Args:
            query: 查询文本
            top_k: 返回的最相似文档数量
            
        Returns:
            包含相似文档信息的列表
        """
        if self.index.ntotal == 0:
            raise ValueError("索引为空，请先调用index_documents()进行索引")
        
        # 计算查询的嵌入向量
        query_vector = self.embedder.embed(query, normalize=True)
        
        # 搜索最相似的向量
        scores, indices = self.index.search(np.array([query_vector]), k=min(top_k, self.index.ntotal))
        
        # 将结果整理为字典
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # -1表示无效结果
                results.append({
                    "corpus_id": int(idx),
                    "score": float(score),
                    "text": self.corpus[idx]
                })
        
        return results
    
    def save(self, filepath: str):
        """
        保存索引到文件
        
        Args:
            filepath: 文件路径
        """
        if self.index is not None:
            faiss.write_index(self.index, filepath)
            logger.info(f"索引已保存到: {filepath}")
    
    def load(self, filepath: str):
        """
        从文件加载索引
        
        Args:
            filepath: 文件路径
        """
        if os.path.exists(filepath):
            self.index = faiss.read_index(filepath)
            logger.info(f"索引已从 {filepath} 加载，共 {self.index.ntotal} 个向量")
        else:
            raise FileNotFoundError(f"索引文件不存在: {filepath}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="日语嵌入模型工具")
    
    # 子命令
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # 嵌入命令
    embed_parser = subparsers.add_parser("embed", help="计算文本嵌入")
    embed_parser.add_argument("--text", type=str, required=True, help="要嵌入的文本")
    embed_parser.add_argument("--model", type=str, default="cl-nagoya/ruri-base", help="嵌入模型名称或路径")
    embed_parser.add_argument("--output", type=str, help="输出文件路径")
    
    # 相似度命令
    sim_parser = subparsers.add_parser("similarity", help="计算文本相似度")
    sim_parser.add_argument("--text1", type=str, required=True, help="第一个文本")
    sim_parser.add_argument("--text2", type=str, required=True, help="第二个文本")
    sim_parser.add_argument("--model", type=str, default="cl-nagoya/ruri-base", help="嵌入模型名称或路径")
    
    # 搜索命令
    search_parser = subparsers.add_parser("search", help="执行语义搜索")
    search_parser.add_argument("--query", type=str, required=True, help="查询文本")
    search_parser.add_argument("--corpus", type=str, required=True, help="语料库文件路径（每行一个文档）")
    search_parser.add_argument("--model", type=str, default="cl-nagoya/ruri-base", help="嵌入模型名称或路径")
    search_parser.add_argument("--top-k", type=int, default=5, help="返回的最相似文档数量")
    search_parser.add_argument("--index-type", type=str, choices=["Flat", "IVF", "HNSW"], default="Flat", help="FAISS索引类型")
    search_parser.add_argument("--save-index", type=str, help="保存索引的文件路径")
    search_parser.add_argument("--load-index", type=str, help="加载索引的文件路径")
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    if args.command == "embed":
        # 创建嵌入模型
        embedder = JapaneseEmbedder(model_name_or_path=args.model)
        
        # 计算嵌入
        embedding = embedder.embed(args.text)
        
        if args.output:
            # 保存嵌入到文件
            np.save(args.output, embedding)
            logger.info(f"嵌入已保存到: {args.output}")
        else:
            # 输出嵌入
            logger.info(f"嵌入: {embedding}")
            logger.info(f"嵌入维度: {embedding.shape}")
    
    elif args.command == "similarity":
        # 创建嵌入模型
        embedder = JapaneseEmbedder(model_name_or_path=args.model)
        
        # 计算相似度
        sim_score = embedder.similarity(args.text1, args.text2)
        
        logger.info(f"文本1: {args.text1}")
        logger.info(f"文本2: {args.text2}")
        logger.info(f"相似度: {sim_score:.4f}")
    
    elif args.command == "search":
        # 创建嵌入模型
        embedder = JapaneseEmbedder(model_name_or_path=args.model)
        
        # 创建索引器
        indexer = FaissIndexer(embedder, index_type=args.index_type)
        
        if args.load_index:
            # 从文件加载索引
            indexer.load(args.load_index)
            
            # 加载语料库
            with open(args.corpus, "r", encoding="utf-8") as f:
                indexer.corpus = [line.strip() for line in f]
        else:
            # 加载语料库
            with open(args.corpus, "r", encoding="utf-8") as f:
                corpus = [line.strip() for line in f]
            
            # 对语料库进行索引
            indexer.index_documents(corpus)
            
            # 保存索引
            if args.save_index:
                indexer.save(args.save_index)
        
        # 执行搜索
        results = indexer.search(args.query, top_k=args.top_k)
        
        # 输出结果
        logger.info(f"查询: {args.query}")
        logger.info(f"找到 {len(results)} 个结果:")
        for i, result in enumerate(results, 1):
            logger.info(f"{i}. 分数: {result['score']:.4f}")
            logger.info(f"   文本: {result['text']}")
    
    else:
        logger.error("请指定子命令: embed, similarity, search")
        return 1

if __name__ == "__main__":
    main() 