#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日语提示扩展工具
提供使用LLM进行日语提示扩展的功能和实用工具
"""

import os
import argparse
import logging
import json
import time
from typing import List, Dict, Union, Optional, Tuple, Any, Callable
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline
import re

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 定义提示模板
# 查询扩展模板
QUERY_EXPANSION_TEMPLATE = """
あなたは日本語の検索クエリを改善するエキスパートです。
与えられたクエリを分析し、検索エンジンの結果を改善するために、元のクエリの意図を保ちながら、複数の代替クエリを生成してください。

元のクエリ: {query}

以下のフォーマットで、3〜5個の代替クエリを生成してください:
1. [代替クエリ1]
2. [代替クエリ2]
3. [代替クエリ3]
...

代替クエリは、元のクエリの意図を様々な表現で表したものにしてください。長さや具体性の異なるクエリを含めると良いでしょう。
"""

# プロンプト改善テンプレート
PROMPT_IMPROVEMENT_TEMPLATE = """
あなたは日本語のLLMプロンプトを改善するエキスパートです。
与えられたプロンプトをより明確で、効果的なものに改善してください。

元のプロンプト:
{prompt}

以下の点に注意して改善してください:
1. 指示を明確にする
2. 適切な文脈を提供する
3. 目的を明確に伝える
4. 具体的な出力形式を指定する（必要な場合）
5. 適切な例を含める（必要な場合）

改善されたプロンプト：
"""

# プロンプト変数テンプレート
PROMPT_VARIABLES_TEMPLATE = """
あなたは日本語のLLMプロンプトエンジニアリングのエキスパートです。
以下のプロンプトテンプレートを分析し、テンプレート内の変数（プレースホルダー）を特定して、変数の例を提案してください。

プロンプトテンプレート:
{prompt}

出力形式:
1. [変数名1]: [説明] - 例: "[例1]", "[例2]"
2. [変数名2]: [説明] - 例: "[例1]", "[例2]"
...

もしプロンプト内に変数がない場合は、追加すべき変数を提案してください。
"""

# プロンプトバリエーションテンプレート
PROMPT_VARIATIONS_TEMPLATE = """
あなたは日本語のLLMプロンプトのバリエーションを生成するエキスパートです。
与えられたプロンプトの複数のバリエーションを生成してください。各バリエーションは、元のプロンプトの意図を維持しながら、異なる表現や構造を持つようにしてください。

元のプロンプト:
{prompt}

以下のフォーマットで、3つのバリエーションを生成してください:

バリエーション1:
[プロンプトのバリエーション1]

バリエーション2:
[プロンプトのバリエーション2]

バリエーション3:
[プロンプトのバリエーション3]
"""

# Evol-Instruct風テンプレート（難しいプロンプト生成）
EVOL_INSTRUCT_TEMPLATE = """
あなたはプロンプトエンジニアリングのエキスパートです。与えられた基本プロンプトを分析し、より複雑で難しいバージョンに進化させてください。

元のプロンプト:
{prompt}

次のステップに従ってプロンプトを進化させてください:

1. 元のプロンプトの主要な意図と目的を特定する
2. 以下の方法でプロンプトを複雑化する:
   - より詳細な指示を追加する
   - より高度な専門知識を要求する要素を追加する
   - 複数のサブタスクを含める
   - 制約条件を追加する（例：特定の単語を使用しない、特定の視点から回答するなど）
   - 様々な条件下でのケースを考慮するよう指示する

進化したプロンプト:
"""

class JapanesePromptExpander:
    """日本語プロンプト拡張器のベースクラス"""
    
    def __init__(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        use_fp16: bool = False,
        temperature: float = 0.7,
        max_new_tokens: int = 1024
    ):
        """
        初期化

        Args:
            model_name_or_path: 使用するモデル名またはパス
            cache_dir: キャッシュディレクトリ
            device: デバイス ('cpu' または 'cuda')
            use_fp16: FP16精度を使用するかどうか（メモリ節約）
            temperature: 生成時の温度パラメータ
            max_new_tokens: 生成する最大トークン数
        """
        self.model_name = model_name_or_path
        self.cache_dir = cache_dir
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        
        # 装置の確認
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # FP16精度
        self.use_fp16 = use_fp16 and self.device == "cuda"
        
        # モデル初期化
        self._initialize_model()
    
    def _initialize_model(self):
        """モデルとトークナイザーを初期化する"""
        pass
    
    def expand_query(self, query: str) -> List[str]:
        """
        クエリを拡張する

        Args:
            query: 拡張する検索クエリ

        Returns:
            拡張されたクエリのリスト
        """
        pass
    
    def improve_prompt(self, prompt: str) -> str:
        """
        プロンプトを改善する

        Args:
            prompt: 改善するプロンプト

        Returns:
            改善されたプロンプト
        """
        pass
    
    def get_prompt_variables(self, prompt: str) -> Dict[str, List[str]]:
        """
        プロンプト内の変数を特定し、例を提案する

        Args:
            prompt: 分析するプロンプトテンプレート

        Returns:
            変数名と例のリストを含む辞書
        """
        pass
    
    def generate_prompt_variations(self, prompt: str, num_variations: int = 3) -> List[str]:
        """
        プロンプトのバリエーションを生成する

        Args:
            prompt: 元のプロンプト
            num_variations: 生成するバリエーション数

        Returns:
            プロンプトのバリエーションのリスト
        """
        pass
    
    def evolve_prompt(self, prompt: str) -> str:
        """
        プロンプトをEvol-Instruct手法で進化させる

        Args:
            prompt: 元のプロンプト

        Returns:
            進化したプロンプト
        """
        pass
    
    def _parse_numbered_list(self, text: str) -> List[str]:
        """
        番号付きリストをパースする

        Args:
            text: パースするテキスト

        Returns:
            リストの項目
        """
        # 改行で分割し、番号付きリストの項目を抽出
        items = []
        for line in text.split("\n"):
            line = line.strip()
            # "1. "、"2. " などのパターンを検索
            match = re.match(r'^\d+\.\s+(.+)$', line)
            if match:
                items.append(match.group(1))
        return items

class LocalJapanesePromptExpander(JapanesePromptExpander):
    """ローカルモデルを使用した日本語プロンプト拡張器"""
    
    def _initialize_model(self):
        """モデルとトークナイザーを初期化する"""
        logger.info(f"モデルを初期化中: {self.model_name}")
        
        # キャッシュディレクトリの確認
        model_path = self.model_name
        if self.cache_dir is not None:
            local_path = os.path.join(self.cache_dir, self.model_name.split("/")[-1])
            if os.path.exists(local_path):
                model_path = local_path
                logger.info(f"ローカルパスからモデルを読み込み: {model_path}")
        
        # トークナイザー
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        
        # モデル
        dtype = torch.float16 if self.use_fp16 else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map=self.device
        )
        
        # パイプライン
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # LangChain用パイプライン
        self.llm = HuggingFacePipeline(pipeline=self.pipe)
        
        logger.info("モデルの初期化が完了しました")
    
    def expand_query(self, query: str) -> List[str]:
        """クエリを拡張する"""
        prompt = PromptTemplate(
            template=QUERY_EXPANSION_TEMPLATE,
            input_variables=["query"]
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.invoke({"query": query})
        output = result.get('text', '')
        
        # 結果をパースして拡張クエリのリストを取得
        expanded_queries = self._parse_numbered_list(output)
        
        return expanded_queries
    
    def improve_prompt(self, prompt: str) -> str:
        """プロンプトを改善する"""
        prompt_template = PromptTemplate(
            template=PROMPT_IMPROVEMENT_TEMPLATE,
            input_variables=["prompt"]
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        result = chain.invoke({"prompt": prompt})
        improved_prompt = result.get('text', '').strip()
        
        return improved_prompt
    
    def get_prompt_variables(self, prompt: str) -> Dict[str, List[str]]:
        """プロンプト内の変数を特定し、例を提案する"""
        prompt_template = PromptTemplate(
            template=PROMPT_VARIABLES_TEMPLATE,
            input_variables=["prompt"]
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        result = chain.invoke({"prompt": prompt})
        output = result.get('text', '')
        
        # 結果をパースして変数と例を抽出
        variables = {}
        lines = output.split("\n")
        for line in lines:
            # "1. [変数名]: [説明] - 例: [例1], [例2]" の形式を探す
            match = re.match(r'^\d+\.\s+([^:]+):\s+([^-]+)-\s+例:\s+(.+)$', line)
            if match:
                var_name = match.group(1).strip()
                examples_str = match.group(3).strip()
                # "例: "[例1]", "[例2]"" からリストを作成
                examples = re.findall(r'"([^"]+)"', examples_str)
                variables[var_name] = examples
        
        return variables
    
    def generate_prompt_variations(self, prompt: str, num_variations: int = 3) -> List[str]:
        """プロンプトのバリエーションを生成する"""
        prompt_template = PromptTemplate(
            template=PROMPT_VARIATIONS_TEMPLATE,
            input_variables=["prompt"]
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        result = chain.invoke({"prompt": prompt})
        output = result.get('text', '')
        
        # 結果をパースしてバリエーションを抽出
        variations = []
        pattern = r'バリエーション\d+:\n(.*?)(?=\n\nバリエーション\d+:|$)'
        matches = re.finditer(pattern, output, re.DOTALL)
        
        for match in matches:
            variations.append(match.group(1).strip())
        
        return variations[:num_variations]
    
    def evolve_prompt(self, prompt: str) -> str:
        """プロンプトをEvol-Instruct手法で進化させる"""
        prompt_template = PromptTemplate(
            template=EVOL_INSTRUCT_TEMPLATE,
            input_variables=["prompt"]
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        result = chain.invoke({"prompt": prompt})
        evolved_prompt = result.get('text', '').strip()
        
        return evolved_prompt

class PromptTemplateGenerator:
    """プロンプトテンプレート生成器"""
    
    def __init__(self, expander: JapanesePromptExpander):
        """
        初期化

        Args:
            expander: プロンプト拡張器
        """
        self.expander = expander
    
    def generate_data_collection_template(self, task_description: str) -> str:
        """
        データ収集用のプロンプトテンプレートを生成する

        Args:
            task_description: タスクの説明

        Returns:
            データ収集用のプロンプトテンプレート
        """
        template = f"""
以下のタスクに関連するデータを作成するプロンプトを生成してください。

タスク説明:
{task_description}

このタスクに関連するトレーニングデータ作成のためのプロンプトテンプレートを開発してください。
テンプレートには、以下の要素を含めてください：
1. AIアシスタントの役割説明
2. データの形式と要件の説明
3. データが満たすべき品質基準
4. サンプル例（必要に応じて）

生成するデータの各要素には変数を使用して、後でカスタマイズできるようにしてください。
"""
        
        return self.expander.improve_prompt(template)
    
    def generate_rag_prompt_template(self, task_description: str) -> str:
        """
        RAG（検索拡張生成）用のプロンプトテンプレートを生成する

        Args:
            task_description: タスクの説明

        Returns:
            RAG用のプロンプトテンプレート
        """
        template = f"""
以下のタスクに関連するRAG（検索拡張生成）用のプロンプトテンプレートを生成してください。

タスク説明:
{task_description}

テンプレートには、以下の要素を含めてください：
1. AIアシスタントの役割説明
2. 検索された文脈情報の使用方法に関する指示
3. 回答の形式と要件の説明
4. 追加の制約条件（必要に応じて）

検索された文脈情報は {{context}} 変数に格納され、ユーザーの質問は {{question}} 変数に格納されることを前提としてください。
"""
        
        return self.expander.improve_prompt(template)
    
    def generate_finetuning_dataset_template(self, task_description: str) -> Dict[str, Any]:
        """
        ファインチューニングデータセット生成用のテンプレートを作成する

        Args:
            task_description: タスクの説明

        Returns:
            テンプレート情報を含む辞書
        """
        template = f"""
以下のタスクに関連するLLMファインチューニング用のデータセット生成プロンプトを作成してください。

タスク説明:
{task_description}

データセット生成のためのテンプレートには、以下の要素を含めてください：
1. 指示（instruction）部分のテンプレート
2. 入力（input）部分のテンプレート
3. 出力（output）部分のテンプレート
4. データ品質基準
5. サンプルデータポイント（3つ程度）

テンプレート内の変数には、データ生成時に適切な値を挿入できるよう、明確な名前を付けてください。
"""
        
        # プロンプトの実行と改善
        improved_template = self.expander.improve_prompt(template)
        
        # テンプレートから変数を抽出
        variables = self.expander.get_prompt_variables(improved_template)
        
        # サンプルバリエーションを生成
        variations = self.expander.generate_prompt_variations(improved_template, num_variations=2)
        
        return {
            "template": improved_template,
            "variables": variables,
            "variations": variations
        }

def parse_args():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(description="日本語プロンプト拡張ツール")
    
    # サブコマンド
    subparsers = parser.add_subparsers(dest="command", help="サブコマンド")
    
    # クエリ拡張コマンド
    expand_parser = subparsers.add_parser("expand", help="検索クエリを拡張する")
    expand_parser.add_argument("--query", type=str, required=True, help="拡張する検索クエリ")
    expand_parser.add_argument("--model", type=str, default="rinna/nekomata-7b-instruction", help="使用するモデル名またはパス")
    expand_parser.add_argument("--output", type=str, help="出力ファイルパス（JSON形式）")
    
    # プロンプト改善コマンド
    improve_parser = subparsers.add_parser("improve", help="プロンプトを改善する")
    improve_parser.add_argument("--prompt", type=str, required=True, help="改善するプロンプト")
    improve_parser.add_argument("--model", type=str, default="rinna/nekomata-7b-instruction", help="使用するモデル名またはパス")
    improve_parser.add_argument("--output", type=str, help="出力ファイルパス")
    
    # 変数抽出コマンド
    vars_parser = subparsers.add_parser("variables", help="プロンプト内の変数を特定する")
    vars_parser.add_argument("--prompt", type=str, required=True, help="分析するプロンプト")
    vars_parser.add_argument("--model", type=str, default="rinna/nekomata-7b-instruction", help="使用するモデル名またはパス")
    vars_parser.add_argument("--output", type=str, help="出力ファイルパス（JSON形式）")
    
    # バリエーション生成コマンド
    var_parser = subparsers.add_parser("variations", help="プロンプトのバリエーションを生成する")
    var_parser.add_argument("--prompt", type=str, required=True, help="元のプロンプト")
    var_parser.add_argument("--model", type=str, default="rinna/nekomata-7b-instruction", help="使用するモデル名またはパス")
    var_parser.add_argument("--num", type=int, default=3, help="生成するバリエーション数")
    var_parser.add_argument("--output", type=str, help="出力ファイルパス（JSON形式）")
    
    # プロンプト進化コマンド
    evolve_parser = subparsers.add_parser("evolve", help="プロンプトを進化させる")
    evolve_parser.add_argument("--prompt", type=str, required=True, help="元のプロンプト")
    evolve_parser.add_argument("--model", type=str, default="rinna/nekomata-7b-instruction", help="使用するモデル名またはパス")
    evolve_parser.add_argument("--output", type=str, help="出力ファイルパス")
    
    # テンプレート生成コマンド
    template_parser = subparsers.add_parser("template", help="特定のタイプのプロンプトテンプレートを生成する")
    template_parser.add_argument("--type", type=str, required=True, choices=["data", "rag", "finetune"], 
                                help="テンプレートタイプ: data=データ収集, rag=検索拡張生成, finetune=ファインチューニング")
    template_parser.add_argument("--description", type=str, required=True, help="タスクの説明")
    template_parser.add_argument("--model", type=str, default="rinna/nekomata-7b-instruction", help="使用するモデル名またはパス")
    template_parser.add_argument("--output", type=str, help="出力ファイルパス")
    
    # 共通オプション
    parser.add_argument("--cache-dir", type=str, help="モデルキャッシュディレクトリ")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], help="使用するデバイス")
    parser.add_argument("--fp16", action="store_true", help="FP16精度を使用する")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成時の温度パラメータ")
    
    return parser.parse_args()

def main():
    """メイン関数"""
    args = parse_args()
    
    # プロンプト拡張器の初期化
    expander = LocalJapanesePromptExpander(
        model_name_or_path=args.model if hasattr(args, 'model') else "rinna/nekomata-7b-instruction",
        cache_dir=args.cache_dir,
        device=args.device,
        use_fp16=args.fp16,
        temperature=args.temperature
    )
    
    # コマンドに基づいて処理
    if args.command == "expand":
        # クエリ拡張
        expanded_queries = expander.expand_query(args.query)
        
        # 結果の出力
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(expanded_queries, f, ensure_ascii=False, indent=2)
            logger.info(f"拡張クエリを保存しました: {args.output}")
        else:
            logger.info(f"元のクエリ: {args.query}")
            logger.info("拡張クエリ:")
            for i, query in enumerate(expanded_queries, 1):
                logger.info(f"{i}. {query}")
    
    elif args.command == "improve":
        # プロンプト改善
        improved_prompt = expander.improve_prompt(args.prompt)
        
        # 結果の出力
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(improved_prompt)
            logger.info(f"改善されたプロンプトを保存しました: {args.output}")
        else:
            logger.info(f"元のプロンプト: {args.prompt}")
            logger.info(f"改善されたプロンプト: {improved_prompt}")
    
    elif args.command == "variables":
        # 変数抽出
        variables = expander.get_prompt_variables(args.prompt)
        
        # 結果の出力
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(variables, f, ensure_ascii=False, indent=2)
            logger.info(f"変数情報を保存しました: {args.output}")
        else:
            logger.info(f"プロンプト: {args.prompt}")
            logger.info("変数:")
            for var_name, examples in variables.items():
                logger.info(f"- {var_name}: 例 = {', '.join(examples)}")
    
    elif args.command == "variations":
        # バリエーション生成
        variations = expander.generate_prompt_variations(args.prompt, num_variations=args.num)
        
        # 結果の出力
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(variations, f, ensure_ascii=False, indent=2)
            logger.info(f"プロンプトバリエーションを保存しました: {args.output}")
        else:
            logger.info(f"元のプロンプト: {args.prompt}")
            logger.info("バリエーション:")
            for i, variation in enumerate(variations, 1):
                logger.info(f"バリエーション {i}:")
                logger.info(variation)
                logger.info("---")
    
    elif args.command == "evolve":
        # プロンプト進化
        evolved_prompt = expander.evolve_prompt(args.prompt)
        
        # 結果の出力
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(evolved_prompt)
            logger.info(f"進化したプロンプトを保存しました: {args.output}")
        else:
            logger.info(f"元のプロンプト: {args.prompt}")
            logger.info(f"進化したプロンプト: {evolved_prompt}")
    
    elif args.command == "template":
        # テンプレート生成器の初期化
        template_generator = PromptTemplateGenerator(expander)
        
        if args.type == "data":
            # データ収集テンプレート
            template = template_generator.generate_data_collection_template(args.description)
            
            # 結果の出力
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(template)
                logger.info(f"データ収集テンプレートを保存しました: {args.output}")
            else:
                logger.info(f"タスク説明: {args.description}")
                logger.info(f"データ収集テンプレート: {template}")
        
        elif args.type == "rag":
            # RAGテンプレート
            template = template_generator.generate_rag_prompt_template(args.description)
            
            # 結果の出力
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(template)
                logger.info(f"RAGテンプレートを保存しました: {args.output}")
            else:
                logger.info(f"タスク説明: {args.description}")
                logger.info(f"RAGテンプレート: {template}")
        
        elif args.type == "finetune":
            # ファインチューニングデータセットテンプレート
            template_info = template_generator.generate_finetuning_dataset_template(args.description)
            
            # 結果の出力
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(template_info, f, ensure_ascii=False, indent=2)
                logger.info(f"ファインチューニングテンプレート情報を保存しました: {args.output}")
            else:
                logger.info(f"タスク説明: {args.description}")
                logger.info(f"テンプレート: {template_info['template']}")
                logger.info("変数:")
                for var_name, examples in template_info['variables'].items():
                    logger.info(f"- {var_name}: 例 = {', '.join(examples)}")
                logger.info("バリエーション:")
                for i, variation in enumerate(template_info['variations'], 1):
                    logger.info(f"バリエーション {i}:")
                    logger.info(variation[:200] + "..." if len(variation) > 200 else variation)
    
    else:
        logger.error("有効なサブコマンドを指定してください: expand, improve, variables, variations, evolve, template")
        return 1

if __name__ == "__main__":
    main() 