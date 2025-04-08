#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日语模型微调工具
提供微调日语模型的功能和实用工具
"""

import os
import argparse
import logging
import json
import math
import torch
import numpy as np
from typing import List, Dict, Union, Optional, Tuple, Any
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig, 
    get_peft_model,
    TaskType
)
import wandb
from tqdm import tqdm

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class JapaneseModelFinetuner:
    """日语模型微调器"""
    
    def __init__(
        self,
        model_name_or_path: str,
        output_dir: str = "./models",
        cache_dir: Optional[str] = None,
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        use_8bit: bool = False,
        use_4bit: bool = True,
        gradient_checkpointing: bool = True,
        bf16: bool = False,
        device_map: str = "auto",
        local_rank: int = 0,
        seed: int = 42
    ):
        """
        初始化日语模型微调器
        
        Args:
            model_name_or_path: 模型名称或路径
            output_dir: 输出目录
            cache_dir: 缓存目录
            use_lora: 是否使用LoRA
            lora_rank: LoRA秩
            lora_alpha: LoRA缩放参数
            lora_dropout: LoRA丢弃率
            use_8bit: 是否使用8位精度
            use_4bit: 是否使用4位精度
            gradient_checkpointing: 是否使用梯度检查点
            bf16: 是否使用BF16精度
            device_map: 设备映射
            local_rank: 本地排名（用于分布式训练）
            seed: 随机种子
        """
        self.model_name_or_path = model_name_or_path
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_8bit = use_8bit
        self.use_4bit = use_4bit
        self.gradient_checkpointing = gradient_checkpointing
        self.bf16 = bf16
        self.device_map = device_map
        self.local_rank = local_rank
        self.seed = seed
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置随机种子
        self._set_seed(seed)
        
        # 加载模型和分词器
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        
        logger.info(f"模型微调器初始化完成，输出目录: {output_dir}")
    
    def _set_seed(self, seed: int):
        """设置随机种子"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
    def _load_model_and_tokenizer(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """加载模型和分词器"""
        logger.info(f"加载模型: {self.model_name_or_path}")
        
        # 确定模型路径
        model_path = self.model_name_or_path
        if self.cache_dir is not None:
            local_path = os.path.join(self.cache_dir, self.model_name_or_path.split("/")[-1])
            if os.path.exists(local_path):
                model_path = local_path
                logger.info(f"从本地路径加载模型: {model_path}")
        
        # 创建量化配置
        if self.use_8bit and self.use_4bit:
            logger.warning("不能同时使用8位和4位量化，将默认使用4位量化")
            self.use_8bit = False
        
        if self.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if self.bf16 else torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        elif self.use_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        else:
            quantization_config = None
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if self.bf16 else torch.float16,
            device_map=self.device_map,
            quantization_config=quantization_config,
            trust_remote_code=True
        )
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # 确保分词器有填充和结束标记
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.pad_token = "[PAD]"
                tokenizer.pad_token_id = 0
        
        # 如果使用LoRA，准备模型
        if self.use_lora:
            logger.info("准备LoRA微调")
            
            # 为K-bit训练准备模型
            if self.use_4bit or self.use_8bit:
                model = prepare_model_for_kbit_training(model)
            
            # 配置LoRA
            target_modules = self._get_target_modules_for_lora(model)
            
            lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules
            )
            
            # 获取PEFT模型
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        
        # 启用梯度检查点以节省显存
        if self.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        return model, tokenizer
    
    def _get_target_modules_for_lora(self, model) -> List[str]:
        """为不同模型确定LoRA的目标模块"""
        # 检查模型架构，为不同架构返回合适的目标模块
        model_config = model.config.to_dict()
        architecture = model_config.get("architectures", [""])[0].lower() if "architectures" in model_config else ""
        model_type = model_config.get("model_type", "").lower()
        
        if any(name in architecture or name in model_type for name in ["llama", "qwen", "mistral"]):
            # Llama, Qwen, Mistral等架构
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif any(name in architecture or name in model_type for name in ["falcon"]):
            # Falcon架构
            return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        elif any(name in architecture or name in model_type for name in ["gpt-neox", "neox"]):
            # GPT-NeoX架构
            return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        elif any(name in architecture or name in model_type for name in ["bert"]):
            # BERT架构
            return ["query", "key", "value", "dense"]
        elif any(name in architecture or name in model_type for name in ["t5"]):
            # T5架构
            return ["q", "k", "v", "o", "wi", "wo"]
        else:
            # 默认目标模块
            logger.warning(f"无法识别模型架构：{architecture or model_type}，使用默认目标模块")
            return ["query_proj", "key_proj", "value_proj", "out_proj", "fc1", "fc2", "lm_head"]
    
    def tokenize_function(self, examples):
        """标记化函数"""
        prompt_input = []
        
        # 根据数据格式构建提示
        for i in range(len(examples["instruction"])):
            instruction = examples["instruction"][i]
            
            if "input" in examples and examples["input"][i]:
                input_text = examples["input"][i]
                prompt = f"{instruction}\n\n{input_text}"
            else:
                prompt = instruction
            
            if "output" in examples:
                output = examples["output"][i]
                full_prompt = f"{prompt}\n\n{output}"
            else:
                full_prompt = prompt
            
            prompt_input.append(full_prompt)
        
        # 标记化
        tokenized_inputs = self.tokenizer(
            prompt_input,
            padding="max_length",
            truncation=True,
            max_length=2048,  # 可以根据需要调整
            return_tensors="pt"
        )
        
        return tokenized_inputs
    
    def tokenize_with_template(self, examples, template="alpaca"):
        """使用指定模板进行标记化"""
        result_input_ids = []
        result_attention_mask = []
        result_labels = []
        
        for i in range(len(examples["instruction"])):
            instruction = examples["instruction"][i]
            input_text = examples["input"][i] if "input" in examples and i < len(examples["input"]) else ""
            output = examples["output"][i] if "output" in examples and i < len(examples["output"]) else ""
            
            # 根据模板格式进行处理
            if template == "alpaca":
                if input_text:
                    prompt = f"### 指示:\n{instruction}\n\n### 输入:\n{input_text}\n\n### 响应:\n"
                else:
                    prompt = f"### 指示:\n{instruction}\n\n### 响应:\n"
                full_prompt = f"{prompt}{output}"
            
            elif template == "llama":
                if input_text:
                    prompt = f"<s>[INST] {instruction}\n\n{input_text} [/INST]"
                else:
                    prompt = f"<s>[INST] {instruction} [/INST]"
                full_prompt = f"{prompt}{output}</s>"
            
            elif template == "chatml":
                if input_text:
                    prompt = f"<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>user\n{instruction}\n\n{input_text}\n<|im_end|>\n<|im_start|>assistant\n"
                else:
                    prompt = f"<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>user\n{instruction}\n<|im_end|>\n<|im_start|>assistant\n"
                full_prompt = f"{prompt}{output}\n<|im_end|>"
            
            else:
                raise ValueError(f"不支持的模板格式: {template}")
            
            # 标记化提示和完整文本
            tokenized_prompt = self.tokenizer(prompt, truncation=False, return_tensors="pt")
            tokenized_full = self.tokenizer(full_prompt, truncation=True, max_length=2048, return_tensors="pt")
            
            # 提取输入ID和注意力掩码
            input_ids = tokenized_full.input_ids[0]
            attention_mask = tokenized_full.attention_mask[0]
            
            # 创建标签，将提示部分设为-100（忽略损失计算）
            prompt_len = tokenized_prompt.input_ids.shape[1]
            labels = input_ids.clone()
            labels[:prompt_len] = -100
            
            # 添加到结果
            result_input_ids.append(input_ids)
            result_attention_mask.append(attention_mask)
            result_labels.append(labels)
        
        # 填充到最大长度
        max_length = max(len(ids) for ids in result_input_ids)
        
        for i in range(len(result_input_ids)):
            padding_length = max_length - len(result_input_ids[i])
            if padding_length > 0:
                # 添加填充
                result_input_ids[i] = torch.cat([result_input_ids[i], torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=torch.long)])
                result_attention_mask[i] = torch.cat([result_attention_mask[i], torch.zeros(padding_length, dtype=torch.long)])
                result_labels[i] = torch.cat([result_labels[i], torch.full((padding_length,), -100, dtype=torch.long)])
        
        return {
            "input_ids": torch.stack(result_input_ids),
            "attention_mask": torch.stack(result_attention_mask),
            "labels": torch.stack(result_labels)
        }
    
    def prepare_dataset(self, dataset_path: str, template: str = "alpaca") -> DatasetDict:
        """准备数据集"""
        logger.info(f"加载数据集: {dataset_path}")
        
        # 尝试加载数据集
        try:
            if os.path.isdir(dataset_path):
                # 从本地目录加载Hugging Face数据集
                dataset = load_from_disk(dataset_path)
            else:
                # 尝试从Hugging Face或文件加载
                dataset = load_dataset(dataset_path)
        except:
            # 尝试从JSON或其他格式加载
            if dataset_path.endswith(".json"):
                with open(dataset_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                dataset = Dataset.from_dict(data)
            elif dataset_path.endswith(".jsonl"):
                dataset = load_dataset("json", data_files=dataset_path)
            else:
                raise ValueError(f"不支持的数据集格式: {dataset_path}")
        
        # 确保数据集有正确的分割
        if isinstance(dataset, Dataset):
            # 单一数据集，创建训练/测试分割
            dataset = dataset.train_test_split(test_size=0.1, seed=self.seed)
        
        # 标记化数据集
        tokenized_dataset = {}
        for split in dataset:
            # 使用带有模板的标记化
            tokenized = dataset[split].map(
                lambda examples: self.tokenize_with_template(examples, template=template),
                batched=True,
                remove_columns=dataset[split].column_names
            )
            tokenized_dataset[split] = tokenized
        
        return DatasetDict(tokenized_dataset)
    
    def train(
        self,
        dataset_path: str,
        template: str = "alpaca",
        batch_size: int = 8,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.03,
        num_train_epochs: int = 3,
        save_steps: int = 100,
        eval_steps: int = 100,
        logging_steps: int = 10,
        max_steps: Optional[int] = None,
        fp16: bool = False,
        bf16: bool = False,
        use_wandb: bool = False,
        wandb_project: str = "japanese-llm-finetuning",
        early_stopping_patience: Optional[int] = None
    ):
        """
        训练模型
        
        Args:
            dataset_path: 数据集路径
            template: 数据模板格式
            batch_size: 批处理大小
            gradient_accumulation_steps: 梯度累积步数
            learning_rate: 学习率
            weight_decay: 权重衰减
            warmup_ratio: 预热比例
            num_train_epochs: 训练轮数
            save_steps: 保存步数
            eval_steps: 评估步数
            logging_steps: 日志记录步数
            max_steps: 最大步数（优先于轮数）
            fp16: 是否使用FP16混合精度
            bf16: 是否使用BF16混合精度
            use_wandb: 是否使用Weights & Biases
            wandb_project: Weights & Biases项目名称
            early_stopping_patience: 早停耐心值（可选）
        """
        # 准备数据集
        tokenized_dataset = self.prepare_dataset(dataset_path, template=template)
        
        # 配置Weights & Biases
        if use_wandb and self.local_rank == 0:
            wandb.init(project=wandb_project, name=f"{self.model_name_or_path.split('/')[-1]}_lora{self.lora_rank}" if self.use_lora else self.model_name_or_path.split('/')[-1])
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_train_epochs=num_train_epochs,
            max_steps=max_steps,
            warmup_ratio=warmup_ratio,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            fp16=fp16 and not bf16,
            bf16=bf16,
            seed=self.seed,
            group_by_length=True,
            report_to=["wandb"] if use_wandb else [],
            ddp_find_unused_parameters=False if self.local_rank != -1 else None,
            disable_tqdm=self.local_rank != 0
        )
        
        # 创建训练器
        callbacks = []
        if early_stopping_patience is not None:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            callbacks=callbacks
        )
        
        # 开始训练
        logger.info("开始训练...")
        trainer.train()
        
        # 保存模型
        if self.local_rank <= 0:
            logger.info("保存模型...")
            trainer.save_model(os.path.join(self.output_dir, "final_model"))
            self.tokenizer.save_pretrained(os.path.join(self.output_dir, "final_model"))
            logger.info(f"模型已保存到: {os.path.join(self.output_dir, 'final_model')}")
        
        # 关闭Weights & Biases
        if use_wandb and self.local_rank == 0:
            wandb.finish()
    
    def export_model(self, output_path: Optional[str] = None):
        """
        导出模型
        
        Args:
            output_path: 输出路径（可选）
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, "exported_model")
        
        os.makedirs(output_path, exist_ok=True)
        
        # 导出模型权重
        if self.use_lora:
            # 如果使用LoRA，只保存LoRA权重
            self.model.save_pretrained(output_path)
        else:
            # 否则保存完整模型
            self.model.save_pretrained(output_path)
        
        # 保存分词器
        self.tokenizer.save_pretrained(output_path)
        
        logger.info(f"模型已导出到: {output_path}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="日语模型微调工具")
    
    # 模型和训练参数
    parser.add_argument("--model", type=str, required=True, help="要微调的模型")
    parser.add_argument("--dataset", type=str, required=True, help="训练数据集路径")
    parser.add_argument("--output-dir", type=str, default="./finetuned_models", help="输出目录")
    parser.add_argument("--cache-dir", type=str, help="缓存目录")
    
    # LoRA参数
    parser.add_argument("--use-lora", action="store_true", help="使用LoRA")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA秩")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA Alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA Dropout")
    
    # 量化参数
    parser.add_argument("--use-8bit", action="store_true", help="使用8位量化")
    parser.add_argument("--use-4bit", action="store_true", help="使用4位量化")
    
    # 训练配置
    parser.add_argument("--gradient-checkpointing", action="store_true", help="使用梯度检查点")
    parser.add_argument("--template", type=str, choices=["alpaca", "llama", "chatml"], default="alpaca", help="数据模板格式")
    parser.add_argument("--batch-size", type=int, default=8, help="批处理大小")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="预热比例")
    parser.add_argument("--num-epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--save-steps", type=int, default=100, help="保存步数")
    parser.add_argument("--eval-steps", type=int, default=100, help="评估步数")
    parser.add_argument("--logging-steps", type=int, default=10, help="日志记录步数")
    parser.add_argument("--max-steps", type=int, help="最大步数（优先于轮数）")
    
    # 精度参数
    parser.add_argument("--fp16", action="store_true", help="使用FP16")
    parser.add_argument("--bf16", action="store_true", help="使用BF16")
    
    # Weights & Biases
    parser.add_argument("--use-wandb", action="store_true", help="使用Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="japanese-llm-finetuning", help="Weights & Biases项目名称")
    
    # 早停参数
    parser.add_argument("--early-stopping-patience", type=int, help="早停耐心值")
    
    # 其他参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--local-rank", type=int, default=0, help="本地排名（用于分布式训练）")
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 创建微调器
    finetuner = JapaneseModelFinetuner(
        model_name_or_path=args.model,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_8bit=args.use_8bit,
        use_4bit=args.use_4bit,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=args.bf16,
        local_rank=args.local_rank,
        seed=args.seed
    )
    
    # 训练模型
    finetuner.train(
        dataset_path=args.dataset,
        template=args.template,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_epochs,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        max_steps=args.max_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        early_stopping_patience=args.early_stopping_patience
    )
    
    # 导出模型
    finetuner.export_model()

if __name__ == "__main__":
    main() 