#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import torch
import transformers
import sentence_transformers
import datasets
import peft
import langchain

print("Python 版本:", sys.version)
print("PyTorch 版本:", torch.__version__)
print("PyTorch CUDA 可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA 设备:", torch.cuda.get_device_name(0))
print("Transformers 版本:", transformers.__version__)
print("Sentence-Transformers 版本:", sentence_transformers.__version__)
print("Datasets 版本:", datasets.__version__)
print("PEFT 版本:", peft.__version__)
print("LangChain 版本:", langchain.__version__)

print("\n环境测试完成，一切正常！")
