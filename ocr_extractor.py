import os
import requests
from pdf2image import convert_from_path
from typing import List
import json
import sys
import time
import base64
from PIL import Image
import io

class MistralOCRExtractor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        # 使用正确的 OCR API 端点
        self.api_url = "https://api.mistral.ai/v1/ocr"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.max_retries = 3
        self.retry_delay = 2  # 秒
        
    def process_image(self, image_path: str) -> str:
        """处理单个图像并返回OCR结果"""
        # 读取图像并转换为base64
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # 构建请求体
        payload = {
            "model": "mistral-ocr-latest",  # 使用专门的OCR模型
            "document": {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_data}"
                }
            },
            "include_image_base64": False,  # 不需要返回base64图像
            "image_limit": 0,  # 不限制图像数量
            "image_min_size": 0  # 不限制最小图像尺寸
        }
        
        print(f"正在处理图像: {image_path}")
        print(f"使用模型: mistral-ocr-latest")
        
        # 添加重试机制
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=60  # 增加超时时间
                )
                
                print(f"API响应状态码: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    # 从响应中提取文本
                    if "pages" in result and len(result["pages"]) > 0:
                        return result["pages"][0].get("markdown", "")
                    return ""
                elif response.status_code == 429:  # 速率限制
                    retry_after = int(response.headers.get("Retry-After", self.retry_delay))
                    print(f"速率限制，等待 {retry_after} 秒后重试...")
                    time.sleep(retry_after)
                    continue
                else:
                    error_msg = f"OCR API调用失败: 状态码 {response.status_code}"
                    try:
                        error_detail = response.json()
                        error_msg += f"\n详细信息: {json.dumps(error_detail, indent=2, ensure_ascii=False)}"
                    except:
                        error_msg += f"\n响应内容: {response.text}"
                    
                    if attempt < self.max_retries - 1:
                        print(f"尝试 {attempt+1}/{self.max_retries} 失败: {error_msg}")
                        print(f"等待 {self.retry_delay} 秒后重试...")
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        raise Exception(error_msg)
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    print(f"请求异常: {str(e)}")
                    print(f"等待 {self.retry_delay} 秒后重试...")
                    time.sleep(self.retry_delay)
                    continue
                else:
                    raise Exception(f"请求失败: {str(e)}")
    
    def process_pdf(self, pdf_path: str, output_path: str):
        """处理PDF文件并保存结果"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
            
        print(f"开始处理PDF: {pdf_path}")
        
        # 将PDF转换为图像
        try:
            images = convert_from_path(pdf_path, dpi=300)
            print(f"PDF已转换为 {len(images)} 页图像")
        except Exception as e:
            raise Exception(f"PDF转换失败: {str(e)}")
        
        # 处理每一页
        extracted_text = []
        for i, image in enumerate(images):
            print(f"正在处理第 {i+1} 页")
            # 保存临时图像文件
            temp_image_path = f"temp_page_{i+1}.png"
            image.save(temp_image_path)
            
            try:
                # 调用Mistral OCR API
                text = self.process_image(temp_image_path)
                page_text = f"## Page {i+1}\n\n{text}\n\n---\n\n"
                extracted_text.append(page_text)
                print(f"第 {i+1} 页处理完成")
            except Exception as e:
                print(f"处理第 {i+1} 页时出错: {str(e)}")
                # 继续处理下一页
                continue
            finally:
                # 清理临时文件
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
        
        # 保存结果
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(extracted_text))
            print(f"结果已保存到: {output_path}")
        except Exception as e:
            raise Exception(f"保存结果失败: {str(e)}")

def main():
    # 从环境变量获取API密钥
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("错误: 未设置MISTRAL_API_KEY环境变量")
        print("请使用以下命令设置API密钥:")
        print("export MISTRAL_API_KEY=\"你的API密钥\"")
        sys.exit(1)
    
    pdf_path = "2021年度 前期課程入学試験問題2次募集.pdf"
    output_path = "output.md"
    
    try:
        extractor = MistralOCRExtractor(api_key)
        extractor.process_pdf(pdf_path, output_path)
        print(f"OCR结果已保存到 {output_path}")
    except Exception as e:
        print(f"程序执行失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 