import requests
import json
import time
import sys
import os
import shutil

# 清除终端
def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

# 获取终端尺寸
def get_terminal_size():
    try:
        columns, rows = shutil.get_terminal_size()
        return columns, rows
    except:
        return 80, 24  # 默认值

# 使用ANSI转义序列控制终端光标
def move_cursor(x, y):
    sys.stdout.write(f"\033[{y};{x}H")
    sys.stdout.flush()

def clear_line():
    sys.stdout.write("\033[2K")  # 清除当前行
    sys.stdout.flush()

def save_cursor_position():
    sys.stdout.write("\033[s")
    sys.stdout.flush()

def restore_cursor_position():
    sys.stdout.write("\033[u")
    sys.stdout.flush()

# 设置滚动区域（从第一行到status_row-1行）
def set_scroll_region(top, bottom):
    sys.stdout.write(f"\033[{top};{bottom}r")
    sys.stdout.flush()

# 使用本地 Ollama 服务回答问题
def ask_ollama(query):
    # 获取终端尺寸
    columns, rows = get_terminal_size()
    
    # 清除屏幕并初始化界面
    clear_terminal()
    
    # 内容区域的开始和结束行
    content_start = 1
    status_row = rows - 2
    
    # 设置滚动区域（从第一行到状态栏之前）
    set_scroll_region(content_start, status_row - 1)
    
    print(f"问题: {query}")
    print("\n回答: ", end="", flush=True)
    
    # 固定状态栏 - 在倒数第二行绘制分隔线
    move_cursor(1, status_row)
    print("-" * columns)
    move_cursor(1, status_row + 1)
    print("准备中...", end="", flush=True)
    
    # 重新移动到回答区域
    answer_row = 3
    answer_col = 7
    move_cursor(answer_col, answer_row)  # 回到"回答: "后面
    
    url = "http://192.168.16.119:11434/api/generate"
    payload = {
        "model": "hf.co/bartowski/Qwen_QWQ-32B-GGUF:Q5_K_S",  # 使用正确的模型名称
        "prompt": query  # 直接使用用户输入的问题作为提示
    }
    response = requests.post(url, json=payload, stream=True)  # 启用流式响应
    full_response = ""
    
    # 用于计算速度的变量
    start_time = time.time()
    last_update_time = start_time
    chunk_count = 0
    total_chars = 0
    
    # 行计数 - 用于处理多行输出
    current_row = answer_row
    current_col = answer_col
    
    try:
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                try:
                    json_data = json.loads(decoded_line)  # 使用 json.loads 解析 JSON
                    if "response" in json_data:
                        response_text = json_data["response"]
                        full_response += response_text
                        
                        # 实时打印响应文本
                        for char in response_text:
                            # 检查是否接近状态栏
                            if current_row >= status_row - 1:
                                # 保存状态栏内容
                                save_cursor_position()
                                
                                # 移动到状态栏并获取其内容
                                move_cursor(1, status_row + 1)
                                status_text = sys.stdout.write("\033[2K")  # 清除状态栏
                                
                                # 执行滚动操作 - 向上滚动一行
                                sys.stdout.write("\033[S")  # 向上滚动屏幕
                                
                                # 恢复状态栏
                                move_cursor(1, status_row)
                                print("-" * columns)  # 重绘分隔线
                                
                                # 返回到输入位置，但向上一行（因为滚动了）
                                current_row -= 1
                                move_cursor(current_col, current_row)
                                
                                # 恢复状态文本
                                restore_cursor_position()
                            
                            # 输出字符
                            sys.stdout.write(char)
                            sys.stdout.flush()
                            current_col += 1
                            
                            # 处理换行或行尾
                            if char == '\n' or current_col >= columns:
                                current_row += 1
                                current_col = 1
                        
                        # 计算速度并更新状态栏
                        current_time = time.time()
                        total_chars += len(response_text)
                        chunk_count += 1
                        
                        # 每0.5秒更新一次速度信息
                        if current_time - last_update_time >= 0.5:
                            elapsed = current_time - start_time
                            chars_per_second = total_chars / elapsed if elapsed > 0 else 0
                            
                            # 保存当前光标位置
                            save_cursor_position()
                            
                            # 移动到状态栏并更新信息
                            move_cursor(1, status_row + 1)
                            clear_line()
                            status_text = f"速度: {chars_per_second:.2f} 字符/秒 | 总字符: {total_chars} | 用时: {elapsed:.2f}秒"
                            sys.stdout.write(status_text)
                            sys.stdout.flush()
                            
                            # 恢复光标位置继续输出
                            restore_cursor_position()
                            
                            last_update_time = current_time
                except json.JSONDecodeError as e:
                    # 保存光标位置
                    save_cursor_position()
                    
                    # 更新状态栏显示错误
                    move_cursor(1, status_row + 1)
                    clear_line()
                    status_text = f"解析错误: {e}"
                    sys.stdout.write(status_text)
                    sys.stdout.flush()
                    
                    # 恢复光标位置
                    restore_cursor_position()
        
        # 最终统计
        total_time = time.time() - start_time
        avg_speed = total_chars / total_time if total_time > 0 else 0
        
        # 更新状态栏显示最终信息
        move_cursor(1, status_row + 1)
        clear_line()
        status_text = f"完成! 总共 {total_chars} 字符, 用时 {total_time:.2f} 秒, 平均速度: {avg_speed:.2f} 字符/秒"
        sys.stdout.write(status_text)
        sys.stdout.flush()
        
        # 恢复滚动区域为整个屏幕
        set_scroll_region(1, rows)
        
        # 移动光标到内容结尾
        move_cursor(1, current_row + 2)
        
        return full_response
    except Exception as e:
        # 恢复滚动区域为整个屏幕
        set_scroll_region(1, rows)
        
        # 显示错误信息
        move_cursor(1, status_row + 1)
        clear_line()
        sys.stdout.write(f"Ollama 请求失败: {e}")
        sys.stdout.flush()
        
        # 移动光标到底部
        move_cursor(1, rows)
        
        raise Exception(f"Ollama 请求失败: {e}")
    finally:
        # 确保恢复滚动区域，无论是否发生异常
        set_scroll_region(1, rows)

# 主程序
if __name__ == "__main__":
    try:
        # 检查是否支持ANSI转义序列
        if os.name == 'nt':
            # Windows需要启用ANSI支持
            os.system('color')
            # 在Windows上可能还需要启用Virtual Terminal Processing
            if sys.platform == 'win32':
                from ctypes import windll, c_int, byref
                stdout_handle = windll.kernel32.GetStdHandle(c_int(-11))
                mode = c_int(0)
                windll.kernel32.GetConsoleMode(stdout_handle, byref(mode))
                mode = c_int(mode.value | 0x0004)
                windll.kernel32.SetConsoleMode(stdout_handle, mode)
            
        while True:
            clear_terminal()
            query = input("请输入你的问题（输入 '退出' 结束）：")
            if query.lower() in ["退出", "exit", "quit"]:
                clear_terminal()
                print("感谢使用，再见！")
                break
            response = ask_ollama(query)
            input("\n按回车继续...")
    except KeyboardInterrupt:
        clear_terminal()
        print("程序已中断，感谢使用！")
    finally:
        # 确保终端状态恢复正常
        set_scroll_region(1, get_terminal_size()[1])