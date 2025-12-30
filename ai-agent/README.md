# AI Agent 智能助手项目

一个基于 Vue 3 + Flask + 通义千问大模型的智能对话系统，提供现代化的聊天界面和强大的AI问答能力。

## 🌟 功能特点

### 前端功能
- 🎨 **现代化聊天界面**：简洁美观的对话界面，支持消息气泡展示
- ⌨️ **实时消息发送**：支持回车键快速发送消息
- ⏳ **加载状态显示**：显示AI思考状态，提升用户体验
- 📱 **响应式设计**：适配桌面和移动设备

### 后端功能
- 🔧 **Flask Web服务**：轻量级高性能的后端框架
- 🤖 **通义千问集成**：通过LangChain集成阿里云通义千问大模型
- 📡 **RESTful API**：提供标准的HTTP接口，方便扩展
- 🔒 **跨域支持**：配置CORS，支持前端跨域请求

## 🛠️ 技术栈

### 前端
- **框架**：Vue 3
- **构建工具**：Vite
- **开发语言**：JavaScript
- **UI组件**：原生CSS（可扩展）

### 后端
- **框架**：Flask 3.1.2
- **大模型集成**：LangChain 1.2.0 + ChatTongyi
- **开发语言**：Python 3.11+
- **依赖管理**：pip

### 模型服务
- **大模型**：通义千问 (qwen-turbo)
- **SDK**：dashscope

## 📦 安装步骤

### 1. 克隆或下载项目

确保您已将项目文件保存到本地目录。

### 2. 安装后端依赖

```bash
# 进入后端目录
cd backend

# 安装依赖
pip install -r requirements.txt
```

### 3. 安装前端依赖

```bash
# 进入前端目录
cd frontend

# 安装依赖
npm install
```

## ⚙️ 配置说明

### 1. 配置通义千问API密钥

在 `backend/app.py` 文件中设置您的通义千问API密钥：

```python
# 配置通义千问模型
chat_model = ChatTongyi(
    api_key="your-api-key",  # 替换为您的API密钥
    model="qwen-turbo"
)
```

> 🔑 **获取API密钥**：
> - 访问阿里云官网：https://www.aliyun.com/
> - 搜索"通义千问"或"DashScope"
> - 注册并获取API密钥

### 2. 服务端口配置

默认端口配置：
- 后端服务：5000
- 前端服务：5173

如需修改端口，请分别修改：
- 后端：`app.run(debug=True, port=5000)`
- 前端：`vite.config.js` 中的 server.port 配置

## 🚀 启动项目

### 方式一：使用批处理脚本（Windows）

项目提供了便捷的启动脚本：

- **仅启动后端**：双击 `backend_start.bat`
- **仅启动前端**：双击 `frontend_start.bat`
- **同时启动前后端**：双击 `start_all.bat`

### 方式二：手动启动

#### 启动后端服务

```bash
cd backend
python app.py
```

后端服务将在 `http://127.0.0.1:5000` 运行

#### 启动前端服务

```bash
cd frontend
npm run dev
```

前端服务将在 `http://localhost:5173` 运行

## 🎯 使用说明

1. **访问应用**：在浏览器中打开 `http://localhost:5173`
2. **开始对话**：在输入框中输入您的问题，点击"发送"或按回车键
3. **查看回复**：AI助手会自动生成回复并显示在聊天界面中

## 📁 项目结构

```
ai-agent/
├── backend/                  # 后端目录
│   ├── app.py               # Flask应用主文件
│   └── requirements.txt     # Python依赖列表
├── frontend/                 # 前端目录
│   ├── src/                 # 源代码
│   │   ├── App.vue          # 聊天组件
│   │   ├── assets/          # 静态资源
│   │   └── main.js          # 应用入口
│   ├── package.json         # npm配置文件
│   └── vite.config.js       # Vite配置
├── backend_start.bat        # 后端启动脚本
├── frontend_start.bat       # 前端启动脚本
├── start_all.bat            # 一键启动脚本
└── README.md                # 项目说明文档
```

## 📚 API文档

### 聊天接口

```
POST /api/chat
```

**请求参数**：
```json
{
  "message": "您的问题"
}
```

**响应示例**：
```json
{
  "message": "您的问题",
  "response": "AI的回答"
}
```

**错误响应**：
```json
{
  "error": "错误信息"
}
```

## 🔧 开发说明

### 前端开发

```bash
# 安装依赖
npm install

# 启动开发服务器
npm run dev

# 构建生产版本
npm run build
```

### 后端开发

```bash
# 安装依赖
pip install -r requirements.txt

# 启动开发服务器
python app.py
```

## 🐛 常见问题

### 1. 前端无法连接后端

**解决方法**：
- 检查后端服务是否正常运行
- 确认CORS配置是否正确
- 检查API请求地址是否正确

### 2. 模型调用失败

**解决方法**：
- 检查API密钥是否正确
- 确认网络连接是否正常
- 查看后端日志获取详细错误信息

### 3. 启动脚本执行失败

**解决方法**：
- 确保Python和npm已正确安装
- 检查环境变量配置
- 尝试手动启动服务

## 📝 更新日志

### v1.0.0 (2025-12-29)
- ✅ 初始版本发布
- ✅ Vue 3前端聊天界面
- ✅ Flask后端API
- ✅ 通义千问模型集成
- ✅ Windows启动脚本

## 📄 许可证

本项目采用 MIT 许可证，详情请查看 LICENSE 文件。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！

## 📧 联系方式

如有问题或建议，请随时联系项目维护者。

---

**感谢使用 AI Agent 智能助手项目！** 🎉