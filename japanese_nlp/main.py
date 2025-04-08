#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日语NLP工具包主脚本
提供统一的命令行界面来访问所有功能
"""

import os
import argparse
import logging
import sys
import importlib
from typing import List, Dict, Any, Optional

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 模块映射
MODULES = {
    "install": "install_models",
    "embed": "embedding_tools",
    "rerank": "reranking_tools",
    "prompt": "prompt_expansion",
    "dataset": "dataset_generation",
    "finetune": "fine_tuning"
}

def show_logo():
    """显示工具包标志"""
    print("""
    日本語NLPツールキット / Japanese NLP Toolkit
    ============================================
      最新の日本語プロンプト拡張・埋め込み・再ランキングモデル
      Latest Japanese Prompt Expansion, Embedding & Reranking Models
    """)

def show_modules():
    """显示可用模块"""
    print("\n可用模块 / Available Modules:")
    print("----------------------------")
    print("  install  - 安装模型 / Install models")
    print("  embed    - 嵌入工具 / Embedding tools")
    print("  rerank   - 重排序工具 / Reranking tools")
    print("  prompt   - 提示扩展 / Prompt expansion")
    print("  dataset  - 数据集生成 / Dataset generation")
    print("  finetune - 模型微调 / Model fine-tuning")
    print("\n使用 'python main.py <模块名> --help' 查看特定模块的帮助。")
    print("Use 'python main.py <module> --help' to see help for a specific module.\n")

def import_module(module_name: str):
    """导入模块"""
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        logger.error(f"无法导入模块 {module_name}: {e}")
        return None

def run_module(module_name: str, args: List[str]):
    """运行指定模块"""
    if module_name not in MODULES:
        logger.error(f"未知模块: {module_name}")
        show_modules()
        return 1
    
    # 导入模块
    module = import_module(MODULES[module_name])
    if not module:
        return 1
    
    # 运行模块主函数
    try:
        # 解析参数并运行
        old_argv = sys.argv
        sys.argv = [MODULES[module_name] + ".py"] + args
        
        # 执行模块主函数
        result = module.main()
        
        # 恢复参数
        sys.argv = old_argv
        
        return result or 0
    except Exception as e:
        logger.error(f"运行模块 {module_name} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return 1

def check_environment():
    """检查环境"""
    # 检查Python版本
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        logger.warning("推荐使用Python 3.8或更高版本")
    
    # 检查依赖项
    try:
        import torch
        logger.info(f"PyTorch版本: {torch.__version__}")
        logger.info(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA版本: {torch.version.cuda}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        logger.warning("未安装PyTorch")
    
    try:
        import transformers
        logger.info(f"Transformers版本: {transformers.__version__}")
    except ImportError:
        logger.warning("未安装Transformers")
    
    try:
        import sentence_transformers
        logger.info(f"Sentence Transformers版本: {sentence_transformers.__version__}")
    except ImportError:
        logger.warning("未安装Sentence Transformers")
    
    # 检查模型目录
    if os.path.exists("./models"):
        logger.info(f"模型目录存在: {os.path.abspath('./models')}")
        # 列出已安装的模型
        model_count = 0
        for root, dirs, files in os.walk("./models"):
            if "config.json" in files:
                model_count += 1
                logger.info(f"发现模型: {os.path.basename(root)}")
        
        if model_count == 0:
            logger.info("未发现已安装的模型，请使用'python main.py install'安装模型")
    else:
        logger.info("模型目录不存在，请使用'python main.py install'安装模型")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="日语NLP工具包")
    parser.add_argument("module", nargs="?", help="要运行的模块")
    parser.add_argument("--check", action="store_true", help="检查环境")
    args, remaining = parser.parse_known_args()
    
    return args, remaining

def main():
    """主函数"""
    show_logo()
    
    args, remaining = parse_args()
    
    # 检查环境
    if args.check:
        check_environment()
        return 0
    
    # 如果没有指定模块，显示帮助
    if not args.module:
        show_modules()
        return 0
    
    # 运行指定模块
    return run_module(args.module, remaining)

if __name__ == "__main__":
    sys.exit(main()) 