import tkinter as tk
from tkinter import ttk, messagebox
import urllib.request
import urllib.error
import json
from datetime import datetime
import webbrowser

class NewsSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("百度搜索工具")
        self.root.geometry("900x600")
        self.root.resizable(True, True)
        
        # 设置样式
        self.style = ttk.Style()
        self.style.configure('TLabel', font=('Segoe UI', 10))
        self.style.configure('TButton', font=('Segoe UI', 10, 'bold'))
        self.style.configure('Header.TLabel', font=('Segoe UI', 24, 'bold'))
        self.style.configure('Subtitle.TLabel', font=('Segoe UI', 12))
        
        # 创建主框架
        self.main_frame = ttk.Frame(self.root, padding=20)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 头部区域
        self.header_frame = ttk.Frame(self.main_frame)
        self.header_frame.pack(fill=tk.X, pady=(0, 30))
        
        self.title_label = ttk.Label(self.header_frame, text="百度搜索工具", style='Header.TLabel')
        self.title_label.pack()
        
        self.subtitle_label = ttk.Label(self.header_frame, text="基于百度搜索API的实时搜索工具", style='Subtitle.TLabel')
        self.subtitle_label.pack()
        
        # 搜索区域
        self.search_frame = ttk.Frame(self.main_frame)
        self.search_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(self.search_frame, textvariable=self.search_var, font=('Segoe UI', 12))
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.search_entry.bind('<Return>', lambda event: self.search_news())
        
        self.search_button = ttk.Button(self.search_frame, text="搜索", command=self.search_news)
        self.search_button.pack(side=tk.RIGHT)
        
        # 加载状态
        self.loading_label = ttk.Label(self.main_frame, text="正在搜索新闻...")
        self.loading_label.pack(pady=20)
        self.loading_label.pack_forget()  # 初始隐藏
        
        # 结果区域
        self.results_frame = ttk.Frame(self.main_frame)
        self.results_frame.pack(fill=tk.BOTH, expand=True)
        
        # 结果头部
        self.results_header = ttk.Frame(self.results_frame)
        self.results_header.pack(fill=tk.X, pady=(0, 10))
        
        self.results_title = ttk.Label(self.results_header, text="搜索结果", font=('Segoe UI', 16, 'bold'))
        self.results_title.pack(side=tk.LEFT)
        
        self.results_count = ttk.Label(self.results_header, text="")
        self.results_count.pack(side=tk.LEFT, padx=(10, 0))
        
        # 新闻列表容器
        self.news_list_frame = ttk.Frame(self.results_frame)
        self.news_list_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建滚动条和文本框用于显示结果
        self.scrollbar = ttk.Scrollbar(self.news_list_frame)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.results_text = tk.Text(self.news_list_frame, wrap=tk.WORD, yscrollcommand=self.scrollbar.set, 
                                   font=('Segoe UI', 10), bg='#f8fafc')
        self.results_text.pack(fill=tk.BOTH, expand=True)
        self.scrollbar.config(command=self.results_text.yview)
        
        # 禁止文本框编辑
        self.results_text.config(state=tk.DISABLED)
        
        # 绑定鼠标事件处理URL点击
        self.results_text.tag_bind('url', '<Button-1>', self.open_url)
        self.results_text.tag_config('url', foreground='#4facfe', underline=True)
        
    def open_url(self, event):
        # 获取点击位置的URL
        index = self.results_text.index(f"@{event.x},{event.y}")
        ranges = self.results_text.tag_ranges('url')
        
        for i in range(0, len(ranges), 2):
            start = ranges[i]
            end = ranges[i+1]
            if start <= index <= end:
                url = self.results_text.get(start, end).replace('   链接: ', '')
                webbrowser.open(url)
                break
        
    def search_news(self):
        keyword = self.search_var.get().strip()
        if not keyword:
            messagebox.showwarning("警告", "请输入搜索关键词")
            return
        
        # 显示加载状态
        self.loading_label.pack(pady=20)
        self.results_frame.pack_forget()
        self.root.update_idletasks()
        
        try:
            # 使用百度搜索API
            search_url = 'https://qianfan.baidubce.com/v2/ai_search/web_search'
            
            request_body = {
                "messages": [
                    {
                        "role": "user",
                        "content": keyword
                    }
                ],
                "edition": "standard",
                "search_source": "baidu_search_v2",
                "search_recency_filter": "week"
            }
            
            # 发送请求
            data = json.dumps(request_body).encode('utf-8')
            headers = {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer bce-v3/ALTAK-RNgimuEPImPtxauS4AilW/52446004d8edd1c4d9676bdd4a2f4c5f614c9838'
            }
            
            req = urllib.request.Request(search_url, data=data, headers=headers, method='POST')
            
            with urllib.request.urlopen(req) as response:
                if response.status != 200:
                    raise Exception(f"API请求失败: {response.status}")
                
                news_data = json.loads(response.read().decode('utf-8'))
            
            # 处理搜索结果
            search_results = []
            if news_data and news_data.get('references') and isinstance(news_data['references'], list):
                search_results = news_data['references']
            elif news_data and news_data.get('results') and isinstance(news_data['results'], list):
                search_results = news_data['results']
            elif isinstance(news_data, list):
                search_results = news_data
            
            self.display_results(keyword, search_results)
            
        except Exception as e:
            messagebox.showerror("错误", f"搜索新闻时出错: {str(e)}")
        finally:
            # 隐藏加载状态
            self.loading_label.pack_forget()
            self.results_frame.pack(fill=tk.BOTH, expand=True)
    
    def display_results(self, keyword, search_results):
        # 更新结果头部
        self.results_title.config(text=f"搜索结果: '{keyword}'")
        
        if not search_results:
            self.results_count.config(text="没有找到相关新闻")
            # 清空结果
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.config(state=tk.DISABLED)
            return
        
        self.results_count.config(text=f"找到 {len(search_results)} 条相关新闻")
        
        # 清空结果
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        # 显示每条新闻
        for i, item in enumerate(search_results, 1):
            title = item.get('title', '无标题')
            url = item.get('url', '')
            date = item.get('date', '')
            source = item.get('website', item.get('source', '百度搜索'))
            summary = item.get('snippet', item.get('content', ''))
            
            # 格式化日期
            if date:
                try:
                    date_obj = datetime.fromisoformat(date.replace('Z', '+00:00'))
                    formatted_date = date_obj.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    formatted_date = date
            else:
                formatted_date = ''
            
            # 添加新闻条目
            self.results_text.insert(tk.END, f"{i}. {title}\n", 'title')
            if formatted_date:
                self.results_text.insert(tk.END, f"   日期: {formatted_date}\n")
            if source:
                self.results_text.insert(tk.END, f"   来源: {source}\n")
            if summary:
                self.results_text.insert(tk.END, f"   摘要: {summary}\n")
            if url:
                self.results_text.insert(tk.END, f"   链接: {url}\n", 'url')
            self.results_text.insert(tk.END, "\n" + "-"*80 + "\n\n")
        
        # 设置文本样式
        self.results_text.tag_config('title', font=('Segoe UI', 12, 'bold'), foreground='#2d3748')
        
        # 禁止编辑
        self.results_text.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = NewsSearchApp(root)
    root.mainloop()