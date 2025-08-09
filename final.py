import streamlit as st
import subprocess
from pypinyin import pinyin, Style
import spacy
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import certifi
import random
from highlight import diff_highlight

# ================== 配置 & 全局常量 ==================
QWEN_MODEL_NAME = "qwen2.5:14b"
MODEL_TIMEOUT = 180

# ================== 资源加载与缓存 ==================
@st.cache_resource
def connect_mongodb():
    uri = (
        "mongodb://2068432802:lzq520796@"
        "cluster0-shard-00-00.tmy62.mongodb.net:27017,"
        "cluster0-shard-00-01.tmy62.mongodb.net:27017,"
        "cluster0-shard-00-02.tmy62.mongodb.net:27017/"
        "?ssl=true&replicaSet=atlas-j16zyw-shard-0&authSource=admin"
        "&retryWrites=true&w=majority&appName=Cluster0"
    )
    try:
        client = MongoClient(uri, tlsCAFile=certifi.where(), server_api=ServerApi('1'))
        client.admin.command('ping')
        print("成功连接到 MongoDB！")
        return client
    except Exception as e:
        st.error(f"数据库连接失败: {e}")
        return None

@st.cache_resource
def initialize_bert():
    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        print("BERT 模型加载成功。")
        return tokenizer, model
    except Exception as e:
        st.error(f"BERT 模型加载失败: {e}")
        return None, None

@st.cache_resource
def load_spacy_model():
    try:
        nlp = spacy.load("zh_core_web_sm")
        print("Spacy 模型加载成功。")
        return nlp
    except Exception as e:
        st.error(f"Spacy 模型加载失败: {e}")
        return None

# ================== 核心功能函数 ==================
def call_local_qwen(prompt: str) -> str:
    try:
        process = subprocess.run(
            ['ollama', 'run', QWEN_MODEL_NAME],
            input=prompt,
            capture_output=True,
            text=True,
            encoding='utf-8',
            check=True,
            timeout=MODEL_TIMEOUT
        )
        return process.stdout.strip()
    except FileNotFoundError:
        return "[本地模型调用失败] 'ollama' 命令未找到。"
    except subprocess.CalledProcessError as e:
        return f"[本地模型调用失败] {e.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return f"[本地模型调用超时] {MODEL_TIMEOUT} 秒内未返回。"
    except Exception as e:
        return f"[本地模型调用出错] {str(e)}"

def get_embedding(text, tokenizer, model):
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state[:, 0, :]

def find_most_similar(input_text, collection, tokenizer, model):
    input_embedding = get_embedding(input_text, tokenizer, model)
    max_similarity = -1
    most_similar_article = None
    for doc in collection.find().limit(100):
        if 'content' not in doc or not doc['content']:
            continue
        db_embedding = get_embedding(doc['content'], tokenizer, model)
        similarity = F.cosine_similarity(input_embedding, db_embedding).item()
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_article = doc
    return most_similar_article, max_similarity

def find_random_article(collection):
    pipeline = [{"$match": {"content": {"$exists": True}}}, {"$sample": {"size": 1}}]
    try:
        result = list(collection.aggregate(pipeline))
        return result[0] if result else None
    except Exception:
        docs = list(collection.find({"content": {"$exists": True}}))
        return random.choice(docs) if docs else None

def adjust_writing_style_local(input_text, reference_text):
    prompt = (
        f"你是一名学术写作专家，擅长根据参考论文调整文本风格。请尽最大可能保持原文内容，不要增加原文没有的内容，只修改写作风格，但修改用词、句式和结构以尽量匹配参考论文的学术风格。只借鉴参考论文的学术风格，不借鉴参考论文的内容。\n参考论文片段：\n{reference_text[:2000]}...\n\n请修改以下文章使其符合参考论文的学术风格：\n{input_text}")
    return call_local_qwen(prompt)

def get_pinyin_with_tone(text):
    pinyin_list = pinyin(text, style=Style.TONE3)
    return " ".join([item[0] for item in pinyin_list])

def generate_syntax_analysis(text, nlp_model):
    doc = nlp_model(text)
    lines = ["[依存句法分析]"]
    for token in doc:
        lines.append(f"词语: {token.text:<5} 词性: {token.pos_:<5} 依存关系: {token.dep_:<10} 支配词: {token.head.text}")
    lines.append("\n[命名实体识别]")
    if not doc.ents:
        lines.append("未识别到命名实体")
    else:
        for ent in doc.ents:
            lines.append(f"实体: {ent.text:<15} 类型: {ent.label_:<10} 位置: ({ent.start_char}-{ent.end_char})")
    return "\n".join(lines)

# ================== Streamlit 主程序 ==================
def main():
    st.set_page_config(page_title="学术写作智能助手", layout="wide")
    st.markdown("<h1 style='text-align:center; color:#4A90E2;'>学术写作智能助手</h1>", unsafe_allow_html=True)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "just_updated" not in st.session_state:
        st.session_state.just_updated = False

    if "mongodb_client" not in st.session_state:
        st.session_state.mongodb_client = connect_mongodb()
    if "bert_tokenizer" not in st.session_state:
        st.session_state.bert_tokenizer, st.session_state.bert_model = initialize_bert()
    if "spacy_model" not in st.session_state:
        st.session_state.spacy_model = load_spacy_model()

    if not all([st.session_state.mongodb_client, st.session_state.bert_tokenizer, st.session_state.spacy_model]):
        st.warning("核心组件加载失败，请检查终端日志。")
        st.stop()

    with st.sidebar:
        st.title("功能选择")
        feature = st.radio("请选择功能", ["文本纠错", "风格迁移", "个性化建议"], key="feature_selection")
        if feature == "风格迁移":
            style_transfer_mode = st.radio("风格迁移模式", ["自动匹配范文", "手动提供范文"], key="style_transfer_mode")
        enable_self_reflection = st.toggle("自我反思功能", value=True) if feature == "文本纠错" else False
        st.markdown("---")
        st.subheader("聊天历史")
        for i, chat in enumerate(st.session_state.chat_history):
            st.markdown(f"**{i + 1}. [{chat['type']}]** {chat['input'][:20]}...")

    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat["input"])
        output = chat.get("highlight_html") or chat.get("corrected_output") or chat.get("adjusted_output") or chat.get("suggestions") or ""
        with st.chat_message("assistant"):
            if chat.get("highlight_html"):
                st.markdown(chat["highlight_html"], unsafe_allow_html=True)
            else:
                st.markdown(output)

    if st.session_state.feature_selection == "风格迁移" and hasattr(st.session_state, "style_transfer_mode") and st.session_state.style_transfer_mode == "手动提供范文":
        if "reference_text" not in st.session_state:
            if reference_text := st.chat_input("范文：(请输入两次，第一次输入范文，第二次输入任意字符）"):
                st.session_state.reference_text = reference_text
                st.session_state.chat_history.append({"type": "风格迁移", "input": f"范文：{reference_text}"})
        else:
            if input_text := st.chat_input("待修改文章："):
                st.session_state.chat_history.append({"type": "风格迁移", "input": f"待修改文章：{input_text}"})
                with st.spinner("AI 正在处理，请稍候..."):
                    adjusted_similar = adjust_writing_style_local(input_text, st.session_state.reference_text)
                    output_message = f"**根据您提供的范文优化后：**\n\n---\n\n{adjusted_similar}"
                    st.session_state.chat_history[-1]["adjusted_output"] = output_message
                    del st.session_state.reference_text
                st.session_state.just_updated = True
    else:
        if input_text := st.chat_input("请输入您要处理的文本..."):
            st.session_state.chat_history.append({"type": st.session_state.feature_selection, "input": input_text})
            with st.spinner("AI 正在处理，请稍候..."):
                feature = st.session_state.feature_selection
                if feature == "文本纠错":
                    pinyin_info = get_pinyin_with_tone(input_text)
                    spelling_prompt = f"你是中文拼写纠错专家，不需要判断文本内容是否合理，而是根据拼音信息，判断并纠正中文文本中可能存在的拼写错误..."
                    spelling_result = call_local_qwen(spelling_prompt)
                    highlight_html = diff_highlight(input_text, spelling_result)
                    st.session_state.chat_history[-1]["corrected_output"] = spelling_result
                    st.session_state.chat_history[-1]["highlight_html"] = highlight_html
                elif feature == "风格迁移":
                    db = st.session_state.mongodb_client.get_database("paper")
                    collection = db.get_collection("paper_segments")
                    ref_doc, sim_score = find_most_similar(input_text, collection, st.session_state.bert_tokenizer, st.session_state.bert_model)
                    if ref_doc:
                        adjusted_similar = adjust_writing_style_local(input_text, ref_doc.get('content', ''))
                        output_message = f"**根据风格相似度最高的三篇论文（最高相似度: {sim_score:.4f}）优化后：**\n\n---\n\n{adjusted_similar}"
                    else:
                        output_message = "抱歉，数据库中未能找到相似的参考论文。"
                    st.session_state.chat_history[-1]["adjusted_output"] = output_message
                elif feature == "个性化建议":
                    suggestions = "这里生成个性化建议..."
                    st.session_state.chat_history[-1]["suggestions"] = suggestions
            st.session_state.just_updated = True

    if st.session_state.just_updated:
        st.session_state.just_updated = False
        st.rerun()

if __name__ == "__main__":
    main()
