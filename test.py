from zhipuai import ZhipuAI
client = ZhipuAI(api_key="420bda55237d4101a6727f4b0d1f036b.MiXxo0k6Ocq9iUqc")

response = client.images.generations(
    model="cogView-4-250304", #填写需要调用的模型编码
    prompt="精美的二次元风格对话框UI设计，融合《美少女梦工厂4》的清新线条与《心跳回忆4》的精致色彩。"
"对话框采用简化的巴洛克式装饰，以平面化的金色线描勾勒华丽边框，避免过度写实质感。"
"左侧为角色头像区，边框左上角有角色名字，右侧为文字区，整体保持动漫游戏的扁平化审美与色彩亮度。"
"边框设计参考二次元游戏中的华丽UI，使用细腻的金色线条勾勒出卷草纹样，而非立体浮雕效果。"
"背景使用渐变的宝蓝与玫瑰色调，采用动漫中常见的平滑渐变而非真实织物纹理。"
"四角点缀简化的花卉装饰，线条干净利落，保持二次元风格的简洁与识别度。"
"整体画面明亮通透，避免过度阴影与写实光影，强调扁平化设计与鲜明色彩对比。"
"金色描边细腻但不过分复杂，保持二次元游戏界面的清晰度与可读性。"
"背景可加入轻微的发光效果和简化的粒子元素，但避免过于写实的光线折射和粒子系统。"
"整体风格偏向高级感的二次元美术，而非3D写实渲染，确保符合动漫游戏的视觉语言。，融合现代二次元UI的简洁与古典美学的优雅。"
"high quality, detailed anime illustration, anime style, flat color, cel shading",
    size="1440x720"
)
import requests
import os

# 获取图片URL
image_url = response.data[0].url

# 下载图片
response = requests.get(image_url)
if response.status_code == 200:
    # 保存到当前目录
    with open("generated_image.png", "wb") as file:
        file.write(response.content)
    print("图片已成功下载到当前目录: generated_image.png")
else:
    print(f"下载失败，状态码: {response.status_code}")

