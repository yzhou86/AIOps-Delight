# OpenSkills 使用指南

## 安装步骤

1. **安装 Node.js**
   访问 [Node.js 官方网站](https://nodejs.org/en/download) 下载并安装适合您操作系统的 Node.js 版本。

2. **安装 OpenSkills**
   打开 PowerShell 终端，执行以下命令全局安装 openskills：
   ```bash
   npm i -g openskills
   ```

3. **创建目录结构**
   在 Trae 中打开您的项目目录，然后创建以下目录结构：
   ```
   .claude\skills
   ```

4. **克隆 Anthropics 官方 Skills 项目**
   执行以下命令克隆官方技能库：
   ```bash
   git clone https://github.com/anthropics/skills.git
   ```

5. **复制技能目录**
   将克隆的 skills 项目中的 `skills\skills\*` 所有目录复制到您项目的 `.claude\skills` 目录下。然后在终端执行同步命令：
   ```bash
   openskills sync
   ```

6. **使用技能**
   在 Trae 的 AI 助手界面，您可以通过对话触发技能调用。例如：
   ```
   create a sample word doc
   ```
   这句话会触发调用 docx 技能来创建一个 Word 文档。