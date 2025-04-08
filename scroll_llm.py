import requests
import json
import time
import sys
import os
import shutil
import threading
import queue
from collections import deque

# 检测操作系统类型，针对不同系统导入不同模块
is_windows = os.name == 'nt'

if not is_windows:
    # Unix系统模块
    import termios
    import tty
    import select
else:
    # Windows系统模块
    import msvcrt

# ======================= 终端控制函数 =======================

def clear_terminal():
    """清除终端内容"""
    os.system('cls' if is_windows else 'clear')

def get_terminal_size():
    """获取终端尺寸"""
    try:
        columns, rows = shutil.get_terminal_size()
        return columns, rows
    except:
        return 80, 24  # 默认值

def move_cursor(x, y):
    """移动光标到指定位置"""
    sys.stdout.write(f"\033[{y};{x}H")
    sys.stdout.flush()

def clear_line():
    """清除当前行"""
    sys.stdout.write("\033[2K")
    sys.stdout.flush()

def hide_cursor():
    """隐藏光标"""
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()

def show_cursor():
    """显示光标"""
    sys.stdout.write("\033[?25h")
    sys.stdout.flush()

def set_text_color(color_code):
    """设置文本颜色"""
    sys.stdout.write(f"\033[{color_code}m")
    sys.stdout.flush()

def reset_text_format():
    """重置文本格式"""
    sys.stdout.write("\033[0m")
    sys.stdout.flush()

# ======================= 键盘输入处理 =======================

class KeyboardListener:
    """键盘输入监听器，用于捕获键盘事件"""
    
    def __init__(self):
        self.key_queue = queue.Queue()
        self.running = False
        self.thread = None
        
    def start(self):
        """开始监听键盘输入"""
        self.running = True
        self.thread = threading.Thread(target=self._listen_keyboard)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        """停止监听键盘输入"""
        self.running = False
        if self.thread:
            self.thread.join(0.5)
    
    def _listen_keyboard(self):
        """监听键盘输入的线程函数"""
        if not is_windows:
            self._listen_keyboard_unix()
        else:
            self._listen_keyboard_windows()
    
    def _listen_keyboard_unix(self):
        """Unix系统下的键盘监听"""
        # 保存原始终端设置
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            # 设置终端为原始模式，不需要按回车就能获取按键
            tty.setraw(fd)
            
            while self.running:
                # 检查是否有键盘输入
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    
                    # 处理特殊键
                    if key == '\x1b':  # ESC键
                        # 等待可能的后续按键
                        if select.select([sys.stdin], [], [], 0.1)[0]:
                            key2 = sys.stdin.read(1)
                            if key2 == '[':
                                if select.select([sys.stdin], [], [], 0.1)[0]:
                                    key3 = sys.stdin.read(1)
                                    # 箭头键和功能键
                                    if key3 == 'A':  # 上箭头
                                        self.key_queue.put('UP')
                                    elif key3 == 'B':  # 下箭头
                                        self.key_queue.put('DOWN')
                                    elif key3 == '5':  # Page Up (需要读取后面的~)
                                        sys.stdin.read(1)
                                        self.key_queue.put('PGUP')
                                    elif key3 == '6':  # Page Down (需要读取后面的~)
                                        sys.stdin.read(1)
                                        self.key_queue.put('PGDN')
                            else:
                                self.key_queue.put('ESC')
                        else:
                            self.key_queue.put('ESC')
                    elif key == 'q' or key == 'Q':
                        self.key_queue.put('QUIT')
                    elif key == ' ':
                        self.key_queue.put('SPACE')
                    elif key == '\r':
                        self.key_queue.put('ENTER')
                    else:
                        self.key_queue.put(key)
        finally:
            # 恢复终端设置
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    
    def _listen_keyboard_windows(self):
        """Windows系统下的键盘监听"""
        while self.running:
            if msvcrt.kbhit():  # 检查是否有键盘输入
                key = msvcrt.getch()  # 获取按键
                
                # 处理特殊键
                if key == b'\xe0' or key == b'\x00':  # 特殊键前缀
                    key2 = msvcrt.getch()
                    if key2 == b'H':  # 上箭头
                        self.key_queue.put('UP')
                    elif key2 == b'P':  # 下箭头
                        self.key_queue.put('DOWN')
                    elif key2 == b'I':  # Page Up
                        self.key_queue.put('PGUP')
                    elif key2 == b'Q':  # Page Down
                        self.key_queue.put('PGDN')
                elif key == b'\x1b':  # ESC键
                    self.key_queue.put('ESC')
                elif key == b'q' or key == b'Q':
                    self.key_queue.put('QUIT')
                elif key == b' ':
                    self.key_queue.put('SPACE')
                elif key == b'\r':
                    self.key_queue.put('ENTER')
                else:
                    try:
                        # 尝试将字节解码为字符
                        decoded_key = key.decode('utf-8')
                        self.key_queue.put(decoded_key)
                    except:
                        pass  # 忽略无法解码的键
            
            # 短暂休眠以减少CPU使用
            time.sleep(0.01)
    
    def get_key(self, timeout=None):
        """获取按键，非阻塞"""
        try:
            return self.key_queue.get(block=timeout is not None, timeout=timeout)
        except queue.Empty:
            return None

# ======================= 文本缓冲区和视图 =======================

class ScrollableTextView:
    """可滚动的文本视图，管理文本缓冲区和视图渲染"""
    
    def __init__(self, max_buffer_lines=1000):
        self.buffer = deque(maxlen=max_buffer_lines)  # 使用双端队列限制最大行数
        self.current_line = ""  # 当前未完成的行
        self.view_start_line = 0  # 当前视图的起始行索引
        self.columns, self.rows = get_terminal_size()
        self.content_rows = self.rows - 3  # 减去问题行、状态栏和分隔线
        
    def add_text(self, text):
        """添加文本到缓冲区"""
        for char in text:
            if char == '\n':
                # 完成当前行，添加到缓冲区
                self.buffer.append(self.current_line)
                self.current_line = ""
            else:
                self.current_line += char
                # 检查行是否已满
                if len(self.current_line) >= self.columns:
                    self.buffer.append(self.current_line)
                    self.current_line = ""
        
        # 如果当前视图在最底部，则自动滚动
        if self.is_at_bottom():
            self.scroll_to_bottom()
            
    def is_at_bottom(self):
        """检查视图是否在缓冲区底部"""
        return self.view_start_line + self.content_rows >= len(self.buffer)
    
    def scroll_up(self, lines=1):
        """向上滚动视图"""
        self.view_start_line = max(0, self.view_start_line - lines)
        
    def scroll_down(self, lines=1):
        """向下滚动视图"""
        max_start = max(0, len(self.buffer) - self.content_rows)
        self.view_start_line = min(max_start, self.view_start_line + lines)
    
    def scroll_to_bottom(self):
        """滚动到底部"""
        self.view_start_line = max(0, len(self.buffer) - self.content_rows)
    
    def page_up(self):
        """向上翻页"""
        self.scroll_up(self.content_rows - 1)
        
    def page_down(self):
        """向下翻页"""
        self.scroll_down(self.content_rows - 1)
        
    def render(self, start_row):
        """渲染视图内容到屏幕"""
        # 计算要显示的行
        end_line = min(len(self.buffer), self.view_start_line + self.content_rows)
        display_lines = list(self.buffer)[self.view_start_line:end_line]
        
        # 清除内容区域
        for i in range(self.content_rows):
            move_cursor(1, start_row + i)
            clear_line()
        
        # 显示缓冲区内容
        for i, line in enumerate(display_lines):
            move_cursor(1, start_row + i)
            # 如果行太长，截断
            if len(line) > self.columns:
                sys.stdout.write(line[:self.columns-3] + "...")
            else:
                sys.stdout.write(line)
        
        # 显示当前未完成的行
        if self.is_at_bottom() and end_line - self.view_start_line < self.content_rows:
            move_cursor(1, start_row + (end_line - self.view_start_line))
            sys.stdout.write(self.current_line)
            
        sys.stdout.flush()
        
    def update_terminal_size(self):
        """更新终端尺寸"""
        self.columns, self.rows = get_terminal_size()
        self.content_rows = self.rows - 3
        # 确保视图起始行在有效范围内
        max_start = max(0, len(self.buffer) - self.content_rows)
        self.view_start_line = min(max_start, self.view_start_line)

# ======================= 状态栏和界面管理 =======================

class StatusBar:
    """状态栏管理，显示速度等信息"""
    
    def __init__(self):
        self.columns, self.rows = get_terminal_size()
        self.status_row = self.rows - 1
        self.status_text = "准备中..."
        self.scroll_mode = False
        
    def update(self, text):
        """更新状态栏文本"""
        self.status_text = text
        self.render()
        
    def toggle_scroll_mode(self, enabled):
        """切换滚动模式状态"""
        self.scroll_mode = enabled
        self.render()
        
    def render(self):
        """渲染状态栏"""
        # 更新终端尺寸
        self.columns, self.rows = get_terminal_size()
        self.status_row = self.rows - 1
        
        # 绘制分隔线
        move_cursor(1, self.status_row - 1)
        sys.stdout.write("-" * self.columns)
        
        # 绘制状态信息
        move_cursor(1, self.status_row)
        clear_line()
        
        # 如果在滚动模式，添加提示
        if self.scroll_mode:
            set_text_color(7)  # 灰色背景
            sys.stdout.write(" 滚动模式 ")
            reset_text_format()
            sys.stdout.write(" | ")
            sys.stdout.write("↑↓: 滚动 | PgUp/PgDn: 翻页 | Space: 回到底部 | Esc: 退出滚动模式 | ")
            
        sys.stdout.write(self.status_text)
        sys.stdout.flush()
        
    def update_terminal_size(self):
        """更新终端尺寸"""
        self.columns, self.rows = get_terminal_size()
        self.status_row = self.rows - 1

# ======================= Ollama API 集成 =======================

class OllamaClient:
    """Ollama API 客户端，处理模型请求和响应"""
    
    def __init__(self, host="http://192.168.16.119:11434"):
        self.base_url = host
        self.model = "hf.co/bartowski/Qwen_QWQ-32B-GGUF:Q5_K_S"
        
    def generate(self, prompt, on_text_callback=None):
        """生成文本"""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt
        }
        
        total_chars = 0
        start_time = time.time()
        full_response = ""
        
        try:
            response = requests.post(url, json=payload, stream=True)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    try:
                        json_data = json.loads(decoded_line)
                        if "response" in json_data:
                            chunk = json_data["response"]
                            full_response += chunk
                            total_chars += len(chunk)
                            
                            # 回调处理新文本
                            if on_text_callback:
                                elapsed = time.time() - start_time
                                chars_per_second = total_chars / elapsed if elapsed > 0 else 0
                                metrics = {
                                    "total_chars": total_chars,
                                    "elapsed": elapsed,
                                    "chars_per_second": chars_per_second
                                }
                                on_text_callback(chunk, metrics)
                                
                    except json.JSONDecodeError as e:
                        pass
            
            # 最终统计
            total_time = time.time() - start_time
            avg_speed = total_chars / total_time if total_time > 0 else 0
            final_metrics = {
                "total_chars": total_chars,
                "total_time": total_time,
                "avg_speed": avg_speed,
                "completed": True
            }
            
            if on_text_callback:
                on_text_callback("", final_metrics)
                
            return full_response
            
        except Exception as e:
            error_message = f"Ollama API 错误: {str(e)}"
            if on_text_callback:
                on_text_callback(error_message, {"error": True})
            return error_message

# ======================= 应用程序主类 =======================

class ScrollableLLMApp:
    """主应用程序，整合所有组件"""
    
    def __init__(self):
        self.text_view = ScrollableTextView()
        self.status_bar = StatusBar()
        self.keyboard = KeyboardListener()
        self.ollama = OllamaClient()
        self.current_query = ""
        self.running = True
        self.scroll_mode = False
        self.generating = False
    
    def initialize(self):
        """初始化应用程序"""
        # 设置终端
        clear_terminal()
        hide_cursor()
        # 开始键盘监听
        self.keyboard.start()
        
    def cleanup(self):
        """清理资源"""
        self.keyboard.stop()
        show_cursor()
        clear_terminal()
        
    def on_new_text(self, text, metrics):
        """处理新生成的文本"""
        if "error" in metrics:
            self.status_bar.update(f"错误: {text}")
            self.generating = False
            return
            
        if text:
            self.text_view.add_text(text)
            self.render_interface()
            
        # 更新状态栏
        if "completed" in metrics and metrics["completed"]:
            status = f"完成! 总共 {metrics['total_chars']} 字符, 用时 {metrics['total_time']:.2f} 秒, 平均速度: {metrics['avg_speed']:.2f} 字符/秒"
            self.status_bar.update(status)
            self.generating = False
        else:
            status = f"速度: {metrics['chars_per_second']:.2f} 字符/秒 | 总字符: {metrics['total_chars']} | 用时: {metrics['elapsed']:.2f}秒"
            self.status_bar.update(status)
    
    def render_interface(self):
        """渲染整个界面"""
        # 显示问题
        move_cursor(1, 1)
        clear_line()
        sys.stdout.write(f"问题: {self.current_query}")
        
        # 渲染文本内容
        content_start_row = 2
        self.text_view.render(content_start_row)
        
        # 渲染状态栏
        self.status_bar.render()
        
    def handle_input(self):
        """处理用户输入"""
        if self.scroll_mode:
            # 处理滚动模式下的按键
            key = self.keyboard.get_key(timeout=0.1)
            if key:
                if key == 'UP':
                    self.text_view.scroll_up()
                    self.render_interface()
                elif key == 'DOWN':
                    self.text_view.scroll_down()
                    self.render_interface()
                elif key == 'PGUP':
                    self.text_view.page_up()
                    self.render_interface()
                elif key == 'PGDN':
                    self.text_view.page_down()
                    self.render_interface()
                elif key == 'SPACE':
                    self.text_view.scroll_to_bottom()
                    self.render_interface()
                elif key == 'ESC' or key == 'q' or key == 'Q':
                    self.scroll_mode = False
                    self.status_bar.toggle_scroll_mode(False)
                    self.render_interface()
        else:
            if not self.generating:
                # 非生成状态，检查是否切换到滚动模式
                key = self.keyboard.get_key(timeout=0.1)
                if key:
                    if key == 'UP' or key == 'PGUP':
                        # 上箭头或PageUp激活滚动模式
                        self.scroll_mode = True
                        self.status_bar.toggle_scroll_mode(True)
                        self.render_interface()
                    elif key == 'ENTER':
                        # 回车键开始新的查询
                        self.prompt_for_query()
                    elif key == 'q' or key == 'Q' or key == 'ESC':
                        self.running = False
    
    def prompt_for_query(self):
        """提示用户输入查询"""
        # 恢复终端正常状态
        show_cursor()
        clear_terminal()
        
        # 获取用户输入
        query = input("请输入你的问题（输入 '退出' 结束）：")
        if query.lower() in ["退出", "exit", "quit"]:
            self.running = False
            return
            
        # 重置UI并开始生成
        self.current_query = query
        self.generating = True
        
        # 清空并重置视图
        self.text_view = ScrollableTextView()
        clear_terminal()
        hide_cursor()
        
        # 显示问题和准备状态
        self.status_bar.update("准备中...")
        self.render_interface()
        
        # 启动生成线程
        threading.Thread(target=self.generate_response, daemon=True).start()
    
    def generate_response(self):
        """生成回复的线程函数"""
        try:
            self.ollama.generate(self.current_query, self.on_new_text)
        except Exception as e:
            self.status_bar.update(f"生成错误: {str(e)}")
            self.generating = False
    
    def update_terminal_size(self):
        """更新所有组件的终端尺寸"""
        self.text_view.update_terminal_size()
        self.status_bar.update_terminal_size()
    
    def run(self):
        """运行应用程序主循环"""
        try:
            self.initialize()
            self.prompt_for_query()
            
            while self.running:
                # 检查终端尺寸变化
                new_cols, new_rows = get_terminal_size()
                if new_cols != self.text_view.columns or new_rows != self.text_view.rows:
                    self.update_terminal_size()
                    self.render_interface()
                
                # 处理输入
                self.handle_input()
                
                # 短暂休眠以减少CPU使用
                time.sleep(0.05)
                
        finally:
            self.cleanup()

# ======================= 主程序入口 =======================

if __name__ == "__main__":
    try:
        # 启用Windows终端ANSI支持
        if is_windows:
            os.system('color')
            try:
                # 尝试启用Virtual Terminal Processing
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            except Exception as e:
                print(f"注意: 无法完全启用Windows终端功能: {e}")
                print("某些显示功能可能受限。")
        
        # 创建并运行应用
        app = ScrollableLLMApp()
        app.run()
        
    except KeyboardInterrupt:
        print("\n程序已中断，感谢使用！")
    except Exception as e:
        print(f"\n程序出错: {str(e)}")
        # 在开发模式下打印完整错误堆栈
        import traceback
        traceback.print_exc()
    finally:
        # 确保终端恢复正常
        show_cursor()
        print("程序已退出。") 