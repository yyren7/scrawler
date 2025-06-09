#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日语数据集生成工具
提供生成日语微调数据集的功能和实用工具
"""

import os
import argparse
import logging
import json
import csv
import random
import re
from typing import List, Dict, Union, Optional, Tuple, Any, Callable
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict, load_dataset

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class DatasetGenerator:
    """日语数据集生成器基类"""
    
    def __init__(
        self,
        output_dir: str = "./data",
        seed: int = 42,
        max_samples: Optional[int] = None
    ):
        """
        初始化数据集生成器
        
        Args:
            output_dir: 输出目录
            seed: 随机种子
            max_samples: 最大样本数（如果指定）
        """
        self.output_dir = output_dir
        self.seed = seed
        self.max_samples = max_samples
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        
        logger.info(f"数据集生成器初始化完成，输出目录: {output_dir}")
    
    def generate(self) -> Dataset:
        """
        生成数据集的抽象方法
        
        Returns:
            生成的数据集
        """
        raise NotImplementedError("子类必须实现generate方法")
    
    def save(self, dataset: Dataset, name: str, format: str = "json"):
        """
        保存数据集
        
        Args:
            dataset: 要保存的数据集
            name: 数据集名称
            format: 保存格式（'json'或'csv'或'parquet'）
        """
        output_path = os.path.join(self.output_dir, name)
        
        if format == "json":
            # 保存为JSON文件
            output_file = f"{output_path}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(dataset.to_dict(), f, ensure_ascii=False, indent=2)
            logger.info(f"数据集已保存为JSON: {output_file}")
        
        elif format == "csv":
            # 保存为CSV文件
            output_file = f"{output_path}.csv"
            dataset.to_csv(output_file, index=False)
            logger.info(f"数据集已保存为CSV: {output_file}")
        
        elif format == "parquet":
            # 保存为Parquet文件
            output_file = f"{output_path}.parquet"
            dataset.to_parquet(output_file)
            logger.info(f"数据集已保存为Parquet: {output_file}")
        
        elif format == "huggingface":
            # 保存为Hugging Face数据集格式
            dataset.save_to_disk(output_path)
            logger.info(f"数据集已保存为Hugging Face格式: {output_path}")
        
        else:
            raise ValueError(f"不支持的格式: {format}")
    
    def _shuffle_and_limit(self, data: List[Dict]) -> List[Dict]:
        """
        打乱数据并限制样本数
        
        Args:
            data: 要处理的数据列表
            
        Returns:
            处理后的数据列表
        """
        # 打乱数据
        random.shuffle(data)
        
        # 限制样本数
        if self.max_samples is not None and len(data) > self.max_samples:
            data = data[:self.max_samples]
        
        return data

class InstructionDatasetGenerator(DatasetGenerator):
    """指令数据集生成器"""
    
    def __init__(
        self, 
        source_datasets: List[Union[str, Dict[str, Any]]] = None,
        instruction_template: str = "{instruction}",
        input_template: str = "{input}",
        output_template: str = "{output}",
        include_input: bool = True,
        train_test_split: float = 0.1, 
        **kwargs
    ):
        """
        初始化指令数据集生成器
        
        Args:
            source_datasets: 源数据集列表（可以是Hugging Face数据集名称或本地文件路径或数据字典）
            instruction_template: 指令模板
            input_template: 输入模板
            output_template: 输出模板
            include_input: 是否包含输入字段
            train_test_split: 训练/测试集划分比例
            **kwargs: 其他参数传递给基类
        """
        super().__init__(**kwargs)
        self.source_datasets = source_datasets or []
        self.instruction_template = instruction_template
        self.input_template = input_template
        self.output_template = output_template
        self.include_input = include_input
        self.train_test_split = train_test_split
    
    def add_source(self, source: Union[str, Dict[str, Any]]):
        """
        添加源数据集
        
        Args:
            source: 源数据集（可以是Hugging Face数据集名称或本地文件路径或数据字典）
        """
        self.source_datasets.append(source)
    
    def load_dataset(self, source: Union[str, Dict[str, Any]]) -> List[Dict]:
        """
        加载数据集
        
        Args:
            source: 源数据集（可以是Hugging Face数据集名称或本地文件路径或数据字典）
            
        Returns:
            加载的数据列表
        """
        if isinstance(source, dict):
            # 直接使用数据字典
            return [source]
        
        elif isinstance(source, str):
            if os.path.exists(source):
                # 本地文件
                ext = os.path.splitext(source)[1].lower()
                
                if ext == '.json':
                    # JSON文件
                    with open(source, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 处理不同格式的JSON
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict) and 'data' in data:
                        return data['data']
                    else:
                        return [data]
                
                elif ext == '.csv':
                    # CSV文件
                    df = pd.read_csv(source)
                    return df.to_dict('records')
                
                elif ext == '.tsv':
                    # TSV文件
                    df = pd.read_csv(source, sep='\t')
                    return df.to_dict('records')
                
                elif ext == '.parquet':
                    # Parquet文件
                    df = pd.read_parquet(source)
                    return df.to_dict('records')
                
                else:
                    raise ValueError(f"不支持的文件格式: {ext}")
            
            else:
                # 尝试作为Hugging Face数据集加载
                try:
                    dataset = load_dataset(source)
                    
                    # 获取第一个分割的数据
                    split_name = list(dataset.keys())[0]
                    return dataset[split_name].to_dict()
                
                except Exception as e:
                    logger.error(f"无法加载数据集 {source}: {e}")
                    return []
        
        else:
            raise ValueError(f"不支持的源数据类型: {type(source)}")
    
    def format_example(self, example: Dict) -> Dict:
        """
        格式化示例为指令格式
        
        Args:
            example: 示例数据
            
        Returns:
            格式化后的示例
        """
        # 提取字段
        instruction = example.get('instruction', '')
        input_text = example.get('input', '')
        output = example.get('output', '')
        
        # 应用模板
        formatted_instruction = self.instruction_template.format(instruction=instruction)
        
        # 构建最终示例
        formatted_example = {
            "instruction": formatted_instruction,
        }
        
        # 如果包含输入字段
        if self.include_input and input_text:
            formatted_input = self.input_template.format(input=input_text)
            formatted_example["input"] = formatted_input
        
        # 输出字段
        formatted_output = self.output_template.format(output=output)
        formatted_example["output"] = formatted_output
        
        return formatted_example
    
    def generate(self) -> DatasetDict:
        """
        生成指令数据集
        
        Returns:
            包含训练集和测试集的数据集字典
        """
        all_examples = []
        
        # 加载所有源数据集
        for source in self.source_datasets:
            logger.info(f"加载源数据集: {source}")
            examples = self.load_dataset(source)
            all_examples.extend(examples)
        
        # 打乱和限制数据
        all_examples = self._shuffle_and_limit(all_examples)
        
        # 格式化示例
        formatted_examples = []
        for example in tqdm(all_examples, desc="格式化示例"):
            try:
                formatted_example = self.format_example(example)
                formatted_examples.append(formatted_example)
            except Exception as e:
                logger.warning(f"格式化示例时出错: {e}, 跳过此示例")
        
        # 划分训练集和测试集
        num_examples = len(formatted_examples)
        num_test = max(1, int(num_examples * self.train_test_split))
        num_train = num_examples - num_test
        
        train_examples = formatted_examples[:num_train]
        test_examples = formatted_examples[num_train:]
        
        logger.info(f"生成了 {num_train} 个训练示例和 {num_test} 个测试示例")
        
        # 创建数据集
        train_dataset = Dataset.from_dict({
            k: [example.get(k, '') for example in train_examples]
            for k in ['instruction', 'input', 'output']
        })
        
        test_dataset = Dataset.from_dict({
            k: [example.get(k, '') for example in test_examples]
            for k in ['instruction', 'input', 'output']
        })
        
        # 返回数据集字典
        return DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })

class TemplateBasedGenerator(DatasetGenerator):
    """基于模板的数据集生成器"""
    
    def __init__(
        self,
        templates: List[str],
        variables: Dict[str, List[str]],
        output_templates: Optional[List[str]] = None,
        num_samples: int = 1000,
        **kwargs
    ):
        """
        初始化基于模板的数据集生成器
        
        Args:
            templates: 模板列表
            variables: 变量及其可能值的字典
            output_templates: 输出模板列表（可选）
            num_samples: 要生成的样本数量
            **kwargs: 其他参数传递给基类
        """
        super().__init__(**kwargs)
        self.templates = templates
        self.variables = variables
        self.output_templates = output_templates
        self.num_samples = num_samples
    
    def generate(self) -> Dataset:
        """
        生成基于模板的数据集
        
        Returns:
            生成的数据集
        """
        examples = []
        
        # 生成指定数量的样本
        for _ in tqdm(range(self.num_samples), desc="生成样本"):
            # 随机选择一个模板
            template = random.choice(self.templates)
            
            # 为每个变量随机选择一个值
            variable_values = {}
            for var_name, values in self.variables.items():
                variable_values[var_name] = random.choice(values)
            
            # 应用模板
            prompt = template.format(**variable_values)
            
            # 如果有输出模板，也随机选择一个
            if self.output_templates:
                output_template = random.choice(self.output_templates)
                output = output_template.format(**variable_values)
            else:
                output = ""
            
            # 构建示例
            example = {
                "instruction": prompt,
                "output": output
            }
            
            examples.append(example)
        
        # 创建数据集
        dataset = Dataset.from_dict({
            "instruction": [ex["instruction"] for ex in examples],
            "output": [ex["output"] for ex in examples]
        })
        
        return dataset

class DataAugmenter:
    """数据增强器"""
    
    def __init__(self, seed: int = 42):
        """
        初始化数据增强器
        
        Args:
            seed: 随机种子
        """
        self.seed = seed
        random.seed(seed)
    
    def augment(self, dataset: Dataset, methods: List[str], multiplier: int = 2) -> Dataset:
        """
        对数据集应用增强方法
        
        Args:
            dataset: 要增强的数据集
            methods: 要应用的增强方法列表
            multiplier: 数据增强的倍数
            
        Returns:
            增强后的数据集
        """
        augmented_data = []
        original_data = dataset.to_dict()
        
        # 对每个示例应用增强
        for i in range(len(dataset)):
            # 添加原始示例
            example = {key: original_data[key][i] for key in original_data}
            augmented_data.append(example)
            
            # 应用增强方法
            for _ in range(multiplier - 1):  # 已经添加了一个原始示例
                aug_example = example.copy()
                
                for method in methods:
                    if method == "shuffle_sentences":
                        aug_example = self._shuffle_sentences(aug_example)
                    elif method == "synonym_replacement":
                        aug_example = self._synonym_replacement(aug_example)
                    elif method == "random_deletion":
                        aug_example = self._random_deletion(aug_example)
                
                augmented_data.append(aug_example)
        
        # 创建新的数据集
        return Dataset.from_dict({
            key: [example[key] for example in augmented_data]
            for key in original_data
        })
    
    def _shuffle_sentences(self, example: Dict) -> Dict:
        """打乱句子顺序"""
        result = example.copy()
        
        # 处理输入字段
        if "input" in example and example["input"]:
            # 按句子分割
            sentences = re.split(r'(。|！|？)', example["input"])
            pairs = [(sentences[i], sentences[i+1]) for i in range(0, len(sentences)-1, 2) if i+1 < len(sentences)]
            
            if pairs:
                # 打乱句子对
                random.shuffle(pairs)
                # 重新组合
                shuffled_text = ''.join([p[0] + p[1] for p in pairs])
                result["input"] = shuffled_text
        
        return result
    
    def _synonym_replacement(self, example: Dict) -> Dict:
        """简单的同义词替换（日语中较难实现，此处为示例）"""
        # 在实际应用中，这里应该使用一个日语同义词词典
        # 由于日语同义词替换复杂，这里只是简单示例
        return example
    
    def _random_deletion(self, example: Dict) -> Dict:
        """随机删除一些非关键字"""
        result = example.copy()
        
        if "input" in example and example["input"]:
            # 随机删除一些字符（非实际应用，仅为示例）
            words = list(example["input"])
            # 随机保留80%-90%的内容
            keep_prob = random.uniform(0.8, 0.9)
            keep_indices = random.sample(range(len(words)), int(len(words) * keep_prob))
            keep_indices.sort()
            result["input"] = ''.join([words[i] for i in keep_indices])
        
        return result

class FormatConverter:
    """格式转换器"""
    
    @staticmethod
    def convert_to_alpaca(dataset: Dataset) -> Dataset:
        """转换为Alpaca格式"""
        return dataset  # Alpaca格式就是基本的指令格式，不需要转换
    
    @staticmethod
    def convert_to_llama(dataset: Dataset) -> Dataset:
        """转换为Llama指令格式"""
        data_dict = dataset.to_dict()
        
        # 构建Llama格式
        prompts = []
        completions = []
        
        for i in range(len(dataset)):
            instruction = data_dict["instruction"][i]
            input_text = data_dict.get("input", [""] * len(dataset))[i]
            output = data_dict["output"][i]
            
            if input_text:
                prompt = f"<s>[INST] {instruction}\n\n{input_text} [/INST]"
            else:
                prompt = f"<s>[INST] {instruction} [/INST]"
            
            prompts.append(prompt)
            completions.append(f"{output}</s>")
        
        # 创建新的数据集
        return Dataset.from_dict({
            "prompt": prompts,
            "completion": completions
        })
    
    @staticmethod
    def convert_to_jsonl(dataset: Dataset, output_file: str):
        """转换为JSONL格式并保存"""
        data_dict = dataset.to_dict()
        
        with open(output_file, "w", encoding="utf-8") as f:
            for i in range(len(dataset)):
                example = {key: data_dict[key][i] for key in data_dict}
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    @staticmethod
    def convert_to_chatml(dataset: Dataset) -> Dataset:
        """转换为ChatML格式"""
        data_dict = dataset.to_dict()
        
        # 构建ChatML格式
        chatml_messages = []
        
        for i in range(len(dataset)):
            instruction = data_dict["instruction"][i]
            input_text = data_dict.get("input", [""] * len(dataset))[i]
            output = data_dict["output"][i]
            
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            
            # 用户消息
            user_message = instruction
            if input_text:
                user_message += "\n\n" + input_text
            messages.append({"role": "user", "content": user_message})
            
            # 助手消息
            messages.append({"role": "assistant", "content": output})
            
            chatml_messages.append({"messages": messages})
        
        # 创建新的数据集
        return Dataset.from_dict({
            "messages": chatml_messages
        })

def create_japanese_examples():
    """创建一些日语示例数据"""
    return [
        {
            "instruction": "以下の質問に答えてください。",
            "input": "日本の首都はどこですか？",
            "output": "日本の首都は東京です。"
        },
        {
            "instruction": "以下の文章を要約してください。",
            "input": "AIとは、人間の知能を模倣し、通常は人間によって行われる活動を自動化するために設計されたコンピュータシステムを指します。これには、データの分析、パターンの認識、意思決定、言語の理解、視覚的認識など、さまざまな能力が含まれます。",
            "output": "AIは人間の知能を模倣し、データ分析、パターン認識、意思決定、言語理解、視覚認識などの活動を自動化するコンピュータシステムです。"
        },
        {
            "instruction": "次の日本語の文を英語に翻訳してください。",
            "input": "私は日本語を勉強しています。",
            "output": "I am studying Japanese."
        },
        {
            "instruction": "次の単語の意味を説明してください。",
            "input": "量子コンピュータ",
            "output": "量子コンピュータは、量子力学の原理を利用して情報を処理するコンピュータです。従来のコンピュータがビットを使用するのに対し、量子コンピュータは量子ビット（キュービット）を使用し、特定の問題を指数関数的に高速に解決できる可能性があります。"
        },
        {
            "instruction": "以下の文章から5つのキーワードを抽出してください。",
            "input": "機械学習は、コンピュータがデータから学習し、パターンを認識し、予測を行うことができるようにするAIの一分野です。教師あり学習、教師なし学習、強化学習など、さまざまな手法があります。最近では、ディープラーニングが画像認識や自然言語処理などのタスクで大きな成功を収めています。",
            "output": "1. 機械学習\n2. データ\n3. パターン認識\n4. 教師あり学習\n5. ディープラーニング"
        }
    ]

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="日语数据集生成工具")
    
    # 子命令
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # 指令数据集生成命令
    instruction_parser = subparsers.add_parser("instruction", help="生成指令数据集")
    instruction_parser.add_argument("--sources", type=str, nargs="+", help="源数据集文件或路径列表")
    instruction_parser.add_argument("--output-dir", type=str, default="./data", help="输出目录")
    instruction_parser.add_argument("--output-name", type=str, default="japanese_instruction", help="输出文件名")
    instruction_parser.add_argument("--instruction-template", type=str, default="{instruction}", help="指令模板")
    instruction_parser.add_argument("--input-template", type=str, default="{input}", help="输入模板")
    instruction_parser.add_argument("--output-template", type=str, default="{output}", help="输出模板")
    instruction_parser.add_argument("--include-input", action="store_true", help="包含输入字段")
    instruction_parser.add_argument("--train-test-split", type=float, default=0.1, help="训练/测试集划分比例")
    instruction_parser.add_argument("--max-samples", type=int, help="最大样本数")
    instruction_parser.add_argument("--seed", type=int, default=42, help="随机种子")
    instruction_parser.add_argument("--format", type=str, choices=["alpaca", "llama", "chatml", "jsonl"], default="alpaca", help="输出格式")
    
    # 模板数据集生成命令
    template_parser = subparsers.add_parser("template", help="生成基于模板的数据集")
    template_parser.add_argument("--templates-file", type=str, required=True, help="模板文件（JSON格式）")
    template_parser.add_argument("--output-dir", type=str, default="./data", help="输出目录")
    template_parser.add_argument("--output-name", type=str, default="template_dataset", help="输出文件名")
    template_parser.add_argument("--num-samples", type=int, default=1000, help="生成的样本数量")
    template_parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    # 数据增强命令
    augment_parser = subparsers.add_parser("augment", help="数据增强")
    augment_parser.add_argument("--input-file", type=str, required=True, help="输入数据集文件")
    augment_parser.add_argument("--output-dir", type=str, default="./data", help="输出目录")
    augment_parser.add_argument("--output-name", type=str, default="augmented_dataset", help="输出文件名")
    augment_parser.add_argument("--methods", type=str, nargs="+", choices=["shuffle_sentences", "synonym_replacement", "random_deletion"], 
                                default=["shuffle_sentences"], help="增强方法")
    augment_parser.add_argument("--multiplier", type=int, default=2, help="数据增强倍数")
    augment_parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    # 格式转换命令
    convert_parser = subparsers.add_parser("convert", help="格式转换")
    convert_parser.add_argument("--input-file", type=str, required=True, help="输入数据集文件")
    convert_parser.add_argument("--output-dir", type=str, default="./data", help="输出目录")
    convert_parser.add_argument("--output-name", type=str, default="converted_dataset", help="输出文件名")
    convert_parser.add_argument("--format", type=str, choices=["alpaca", "llama", "chatml", "jsonl"], required=True, help="输出格式")
    
    # 示例数据生成命令
    example_parser = subparsers.add_parser("examples", help="生成示例数据")
    example_parser.add_argument("--output-dir", type=str, default="./data", help="输出目录")
    example_parser.add_argument("--output-name", type=str, default="japanese_examples", help="输出文件名")
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    if args.command == "instruction":
        # 创建指令数据集生成器
        generator = InstructionDatasetGenerator(
            source_datasets=args.sources if args.sources else [],
            instruction_template=args.instruction_template,
            input_template=args.input_template,
            output_template=args.output_template,
            include_input=args.include_input,
            train_test_split=args.train_test_split,
            output_dir=args.output_dir,
            seed=args.seed,
            max_samples=args.max_samples
        )
        
        # 如果没有提供源数据集，使用一些示例数据
        if not args.sources:
            logger.info("未提供源数据集，使用示例数据")
            examples = create_japanese_examples()
            for example in examples:
                generator.add_source(example)
        
        # 生成数据集
        dataset = generator.generate()
        
        # 根据指定格式转换
        if args.format == "alpaca":
            # 已经是Alpaca格式，不需要转换
            pass
        elif args.format == "llama":
            dataset_train = FormatConverter.convert_to_llama(dataset["train"])
            dataset_test = FormatConverter.convert_to_llama(dataset["test"])
            dataset = DatasetDict({
                "train": dataset_train,
                "test": dataset_test
            })
        elif args.format == "chatml":
            dataset_train = FormatConverter.convert_to_chatml(dataset["train"])
            dataset_test = FormatConverter.convert_to_chatml(dataset["test"])
            dataset = DatasetDict({
                "train": dataset_train,
                "test": dataset_test
            })
        
        # 保存数据集
        if args.format == "jsonl":
            train_output = os.path.join(args.output_dir, f"{args.output_name}_train.jsonl")
            test_output = os.path.join(args.output_dir, f"{args.output_name}_test.jsonl")
            FormatConverter.convert_to_jsonl(dataset["train"], train_output)
            FormatConverter.convert_to_jsonl(dataset["test"], test_output)
            logger.info(f"训练集已保存为JSONL: {train_output}")
            logger.info(f"测试集已保存为JSONL: {test_output}")
        else:
            generator.save(dataset["train"], f"{args.output_name}_train", "huggingface")
            generator.save(dataset["test"], f"{args.output_name}_test", "huggingface")
    
    elif args.command == "template":
        # 加载模板文件
        with open(args.templates_file, "r", encoding="utf-8") as f:
            template_data = json.load(f)
        
        # 创建基于模板的数据集生成器
        generator = TemplateBasedGenerator(
            templates=template_data.get("templates", []),
            variables=template_data.get("variables", {}),
            output_templates=template_data.get("output_templates", []),
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            seed=args.seed
        )
        
        # 生成数据集
        dataset = generator.generate()
        
        # 保存数据集
        generator.save(dataset, args.output_name, "huggingface")
    
    elif args.command == "augment":
        # 加载输入数据集
        try:
            dataset = load_dataset(args.input_file)
            # 获取第一个分割的数据
            split_name = list(dataset.keys())[0]
            dataset = dataset[split_name]
        except:
            # 尝试直接加载文件
            if args.input_file.endswith(".json"):
                with open(args.input_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                dataset = Dataset.from_dict(data)
            elif args.input_file.endswith(".csv"):
                dataset = Dataset.from_pandas(pd.read_csv(args.input_file))
            else:
                raise ValueError(f"不支持的文件格式: {args.input_file}")
        
        # 创建数据增强器
        augmenter = DataAugmenter(seed=args.seed)
        
        # 应用数据增强
        augmented_dataset = augmenter.augment(
            dataset=dataset,
            methods=args.methods,
            multiplier=args.multiplier
        )
        
        # 保存增强后的数据集
        output_path = os.path.join(args.output_dir, args.output_name)
        augmented_dataset.save_to_disk(output_path)
        logger.info(f"增强后的数据集已保存: {output_path}")
    
    elif args.command == "convert":
        # 加载输入数据集
        try:
            dataset = load_dataset(args.input_file)
            # 获取第一个分割的数据
            split_name = list(dataset.keys())[0]
            dataset = dataset[split_name]
        except:
            # 尝试直接加载文件
            if args.input_file.endswith(".json"):
                with open(args.input_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                dataset = Dataset.from_dict(data)
            elif args.input_file.endswith(".csv"):
                dataset = Dataset.from_pandas(pd.read_csv(args.input_file))
            else:
                raise ValueError(f"不支持的文件格式: {args.input_file}")
        
        # 根据指定格式转换
        if args.format == "alpaca":
            converted_dataset = FormatConverter.convert_to_alpaca(dataset)
        elif args.format == "llama":
            converted_dataset = FormatConverter.convert_to_llama(dataset)
        elif args.format == "chatml":
            converted_dataset = FormatConverter.convert_to_chatml(dataset)
        
        # 保存转换后的数据集
        if args.format == "jsonl":
            output_file = os.path.join(args.output_dir, f"{args.output_name}.jsonl")
            FormatConverter.convert_to_jsonl(dataset, output_file)
            logger.info(f"转换后的数据集已保存为JSONL: {output_file}")
        else:
            output_path = os.path.join(args.output_dir, args.output_name)
            converted_dataset.save_to_disk(output_path)
            logger.info(f"转换后的数据集已保存: {output_path}")
    
    elif args.command == "examples":
        # 创建示例数据
        examples = create_japanese_examples()
        
        # 转换为数据集
        dataset = Dataset.from_dict({
            "instruction": [ex["instruction"] for ex in examples],
            "input": [ex["input"] for ex in examples],
            "output": [ex["output"] for ex in examples]
        })
        
        # 保存示例数据集
        output_path = os.path.join(args.output_dir, args.output_name)
        dataset.save_to_disk(output_path)
        logger.info(f"示例数据集已保存: {output_path}")
    
    else:
        logger.error("请指定子命令: instruction, template, augment, convert, examples")
        return 1

if __name__ == "__main__":
    main() 