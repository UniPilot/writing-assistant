# 学术写作智能助手 📝
！！现在还需下载highlight.py
这是一个基于 Streamlit 构建的中文文本辅助工具，提供如下两个核心功能：

1. **风格迁移**：根据参考论文（MongoDB 存储）调整用户输入文本的写作风格，调用本地 Qwen2.5 模型。
2. **语义纠错**：结合拼音和语法分析，分阶段纠正拼写与语病错误，并进行自反思检查，调用本地 Qwen2.5 模型。

---

## 🚀 功能展示

- 🔍 学术写作风格迁移（基于参考论文）
- ✍️ 拼音与语法双重分析纠错
- 🧠 自反思机制增强纠错质量

---
### 本地化大模型
 访问官网下载安装包：https://ollama.com/download
 
 下载后拉取大模型
 ```bash
ollama pull qwen2.5:14b
```
拉取完成后，使用命令测试
```bash
ollama run qwen2.5:14b
```
你将看到模型等待输入提示，说明部署成功。
# 创建文件
下载final.py以及requirments.txt到空文件夹后

命令行导航到目录文件夹（该空文件夹）

# 创建环境
```bash
python -m venv venv
```
```bansh
.\venv\Scripts\activate
```
可以看到环境被激活（green）
# 安装依赖
```bash
pip install -r requirements.txt
```
下载需要的包
```bash
python -m spacy download zh_core_web_sm
```
# 程序运行
本地环境被激活以及大模型通过命令行调用正常后，终端输入
```bash
streamlit run final.py
```
即可运行程序，并自动导航到浏览器前端页面进行交互

注意：等待时间较长，没报错就是能成功运行
