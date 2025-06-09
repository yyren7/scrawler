# 日语NLP工具包

这个工具包提供了最新的日语提示扩展（Prompt Expansion）、嵌入（Embedding）和重排序（Reranking）模型，以及用于数据集生成和模型微调的实用工具。

## 功能特点

- **模型安装**：自动安装最新的日语NLP模型
- **嵌入工具**：用于文本嵌入的实用工具
- **重排序工具**：用于改进检索结果的重排序功能
- **提示扩展**：通过LLM增强日语提示
- **数据集生成**：为微调创建高质量的日语指令数据集
- **模型微调**：使用LoRA等技术高效微调日语LLM

## 安装

首先确保安装所有必要的依赖：

```bash
pip install -r requirements.txt
```

## 模型安装

使用安装脚本下载和安装模型：

```bash
# 安装默认的基础模型
python install_models.py

# 安装高级选项
python install_models.py --embedding large --reranking large --llm small

# 查看帮助
python install_models.py --help
```

支持的模型选项：

- 嵌入模型：`large`, `base`, `small`, `multilingual`, `legacy`
- 重排序模型：`large`, `base`, `small`, `cross-encoder`
- LLM模型（可选）：`qwen`, `llama`, `open-source`, `small`

## 嵌入工具使用

嵌入工具提供了文本嵌入、相似度计算和语义搜索功能：

```bash
# 计算嵌入
python embedding_tools.py embed --text "このテキストの埋め込みを計算します。" --model cl-nagoya/ruri-base

# 计算相似度
python embedding_tools.py similarity --text1 "こんにちは" --text2 "おはよう" --model cl-nagoya/ruri-base

# 执行语义搜索
python embedding_tools.py search --query "日本の歴史" --corpus corpus.txt --model cl-nagoya/ruri-base --top-k 5

# 查看帮助
python embedding_tools.py --help
```

## 重排序工具使用

重排序工具可以改进检索结果的排序：

```bash
# 预测相关性分数
python reranking_tools.py predict --query "東京観光のおすすめスポット" --passages passages.txt --model cl-nagoya/ruri-reranker-base

# 重排序候选段落
python reranking_tools.py rerank --query "東京観光のおすすめスポット" --passages passages.txt --scores scores.txt --model cl-nagoya/ruri-reranker-base --top-k 10

# 执行检索-重排序流水线
python reranking_tools.py pipeline --query "東京観光のおすすめスポット" --corpus corpus.txt --model cl-nagoya/ruri-reranker-base --tokenize-japanese

# 查看帮助
python reranking_tools.py --help
```

## 提示扩展工具使用

提示扩展工具可以通过LLM增强和改进提示：

```bash
# 扩展搜索查询
python prompt_expansion.py expand --query "東京 観光" --model rinna/nekomata-7b-instruction

# 改进提示
python prompt_expansion.py improve --prompt "日本について説明してください。" --model rinna/nekomata-7b-instruction

# 提取变量
python prompt_expansion.py variables --prompt "次の{topic}について{length}字で説明してください。" --model rinna/nekomata-7b-instruction

# 生成提示变体
python prompt_expansion.py variations --prompt "日本の歴史について説明してください。" --model rinna/nekomata-7b-instruction --num 3

# 进化提示（复杂化）
python prompt_expansion.py evolve --prompt "日本の四季について説明してください。" --model rinna/nekomata-7b-instruction

# 生成模板
python prompt_expansion.py template --type rag --description "東京の観光スポットに関する質問応答システム" --model rinna/nekomata-7b-instruction

# 查看帮助
python prompt_expansion.py --help
```

## 数据集生成工具使用

数据集生成工具可以为微调创建高质量的日语指令数据集：

```bash
# 生成指令数据集
python dataset_generation.py instruction --sources data1.json data2.json --output-dir ./data --output-name japanese_instruction --format alpaca

# 使用模板生成数据集
python dataset_generation.py template --templates-file templates.json --output-dir ./data --num-samples 1000

# 数据增强
python dataset_generation.py augment --input-file dataset.json --output-dir ./data --methods shuffle_sentences random_deletion --multiplier 2

# 转换格式
python dataset_generation.py convert --input-file dataset.json --output-dir ./data --format llama

# 生成示例数据
python dataset_generation.py examples --output-dir ./data

# 查看帮助
python dataset_generation.py --help
```

## 模型微调工具使用

模型微调工具支持使用QLoRA等技术高效微调日语LLM：

```bash
# 使用LoRA微调模型
python fine_tuning.py --model rinna/nekomata-7b-instruction --dataset ./data/japanese_instruction --use-lora --lora-rank 8 --use-4bit --gradient-checkpointing

# 全参数微调
python fine_tuning.py --model rinna/nekomata-7b-instruction --dataset ./data/japanese_instruction --batch-size 4 --gradient-accumulation-steps 8 --num-epochs 3 --bf16

# 使用自定义模板
python fine_tuning.py --model rinna/nekomata-7b-instruction --dataset ./data/japanese_instruction --use-lora --template chatml --use-4bit

# 查看帮助
python fine_tuning.py --help
```

## 使用示例

### 1. 完整的NLP流水线

```bash
# 1. 安装模型
python install_models.py --embedding base --reranking base

# 2. 准备语料库和查询
echo "日本の歴史は縄文時代から始まります。" > corpus.txt
echo "平安時代は日本の古典文学が栄えた時代です。" >> corpus.txt
echo "江戸時代は鎖国政策が実施された時代です。" >> corpus.txt
echo "明治時代には西洋の技術や文化が導入されました。" >> corpus.txt

# 3. 执行嵌入和检索
python embedding_tools.py search --query "日本の歴史" --corpus corpus.txt --model cl-nagoya/ruri-base --top-k 3 --output results.json

# 4. 对结果进行重排序
python reranking_tools.py rerank --query "日本の歴史" --passages corpus.txt --model cl-nagoya/ruri-reranker-base --top-k 2 --output reranked.json

# 5. 使用提示扩展生成更多查询
python prompt_expansion.py expand --query "日本の歴史" --model rinna/nekomata-7b-instruction --output expanded_queries.json
```

### 2. 数据集生成和模型微调

```bash
# 1. 生成数据集
python dataset_generation.py examples --output-dir ./data
python dataset_generation.py augment --input-file ./data/japanese_examples --output-dir ./data --methods shuffle_sentences --multiplier 2 --output-name augmented_examples

# 2. 微调模型
python fine_tuning.py --model rinna/nekomata-7b-instruction --dataset ./data/augmented_examples --use-lora --lora-rank 8 --use-4bit --num-epochs 3 --output-dir ./finetuned --gradient-checkpointing
```

## 模型资源

本工具包使用的主要模型：

- **嵌入模型**：
  - [cl-nagoya/ruri-base](https://huggingface.co/cl-nagoya/ruri-base) - 最新的日语句子嵌入模型
  - [cl-nagoya/ruri-pt-large](https://huggingface.co/cl-nagoya/ruri-pt-large) - 大型日语句子嵌入模型

- **重排序模型**：
  - [cl-nagoya/ruri-reranker-base](https://huggingface.co/cl-nagoya/ruri-reranker-base) - 日语重排序模型
  - [hotchpotch/japanese-reranker-cross-encoder-base-v1](https://huggingface.co/hotchpotch/japanese-reranker-cross-encoder-base-v1) - 日语交叉编码器

- **LLM模型**（用于提示扩展和微调）：
  - [rinna/nekomata-7b-instruction](https://huggingface.co/rinna/nekomata-7b-instruction) - 轻量级日语指令模型
  - [SakanaAI/EvoLLM-JP-v1-7B](https://huggingface.co/SakanaAI/EvoLLM-JP-v1-7B) - 开源日语模型

## 数据格式

### 指令数据集格式

微调数据集应使用以下格式：

```json
{
  "instruction": "指示内容",
  "input": "输入内容（可选）",
  "output": "期望的输出内容"
}
```

### 模板文件格式

模板生成器使用的JSON格式：

```json
{
  "templates": [
    "次の{topic}について{length}文字で説明してください。",
    "{topic}の主な特徴を{num_points}点挙げてください。"
  ],
  "variables": {
    "topic": ["日本の歴史", "人工知能", "気候変動"],
    "length": ["300", "500", "1000"],
    "num_points": ["3", "5", "7"]
  },
  "output_templates": [
    "{topic}は...",
    "{topic}について説明します。..."
  ]
}
```

## 贡献

欢迎贡献代码、添加模型或提出问题。请提交Issue或Pull Request。

## 许可证

MIT 