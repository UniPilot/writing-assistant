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

# ====== 加载中文模型 ======
nlp = spacy.load("zh_core_web_sm")
ENABLE_SELF_REFLECTION = True

# ====== 本地 Qwen 调用函数 ======
def call_local_qwen(prompt: str) -> str:
    try:
        process = subprocess.Popen(
            ['ollama', 'run', 'qwen2.5:14b'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8'
        )
        output, error = process.communicate(prompt)
        if process.returncode != 0:
            return f"[本地模型调用失败] {error.strip()}"
        return output.strip()
    except Exception as e:
        return f"[本地模型调用出错] {str(e)}"

# ====== MongoDB 连接函数 ======
def connect_mongodb():
    uri = "mongodb+srv://2068432802:lzq520796@cluster0.tmy62.mongodb.net/?retryWrites=true&w=majority"
    client = MongoClient(uri,
                         tlsCAFile=certifi.where(),
                         server_api=ServerApi('1'))
    try:
        client.admin.command('ping')
        print("成功连接到 MongoDB!")
        return client
    except Exception as e:
        print("连接失败:", e)
        return None

# ====== BERT 初始化 ======
def initialize_bert():
    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        return tokenizer, model
    except Exception as e:
        print(f"BERT 模型加载失败: {e}")
        return None, None

# ====== 计算文本嵌入 ======
def get_embedding(text, tokenizer, model):
    tokens = tokenizer(text,
                       return_tensors='pt',
                       truncation=True,
                       padding=True,
                       max_length=512)
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state[:, 0, :]

# ====== 相似度搜索 ======
def find_most_similar(input_text, collection, tokenizer, model):
    input_embedding = get_embedding(input_text, tokenizer, model)

    max_similarity = -1
    most_similar_article = None

    for doc in collection.find():
        if 'content' not in doc:
            continue

        db_embedding = get_embedding(doc['content'], tokenizer, model)
        similarity = F.cosine_similarity(input_embedding, db_embedding).item()

        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_article = doc

    return most_similar_article, max_similarity

# ====== 随机选文 ======
def find_random_article(collection):
    docs = list(collection.find({"content": {"$exists": True}}))
    return random.choice(docs) if docs else None

# ====== 风格迁移（调用本地模型） ======
def adjust_writing_style_local(input_text, reference_text):
    prompt = (
        "你是一名学术写作专家，擅长根据参考论文调整文本风格。请尽最大可能保持原文内容，"
        "不要增加原文没有的内容，只修改写作风格，但修改用词、句式和结构以尽量匹配参考论文的学术风格。\n"
        f"参考论文片段：\n{reference_text[:2000]}...\n\n"
        f"请修改以下文章使其符合参考论文的学术风格：\n{input_text}"
    )
    return call_local_qwen(prompt)

# ====== 语法分析 ======
def generate_syntax_analysis(text):
    doc = nlp(text)
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

# ====== 获取拼音 ======
def get_pinyin_with_tone(text):
    pinyin_list = pinyin(text, style=Style.TONE3)
    return " ".join([item[0] for item in pinyin_list])

# ====== Streamlit App ======
def main():
    st.title("📝 中文文本助手")

    # 连接 MongoDB 和 BERT 模型，App启动时只做一次
    if "mongodb_client" not in st.session_state:
        st.session_state.mongodb_client = connect_mongodb()
    if "bert_tokenizer" not in st.session_state or "bert_model" not in st.session_state:
        tokenizer, model = initialize_bert()
        st.session_state.bert_tokenizer = tokenizer
        st.session_state.bert_model = model

    # 功能选项卡
    feature = st.radio("请选择功能", ["风格迁移", "语义纠错"], horizontal=True)

    # 输入框
    user_input = st.text_area(f"请输入文本（当前功能：{feature}）", height=150)

    if st.button("执行"):
        if not user_input.strip():
            st.warning("请输入有效文本！")
            return

        if feature == "风格迁移":
            if not st.session_state.mongodb_client or not st.session_state.bert_tokenizer or not st.session_state.bert_model:
                st.error("数据库或模型未正确初始化")
                return

            db = st.session_state.mongodb_client.get_database("paper")
            collection = db.get_collection("papers")

            # 找相似参考论文
            ref_doc, sim_score = find_most_similar(user_input, collection, st.session_state.bert_tokenizer, st.session_state.bert_model)
            # 找随机参考论文
            random_doc = find_random_article(collection)

            st.write("正在生成风格迁移结果，请稍候...")

            try:
                adjusted_similar = adjust_writing_style_local(user_input, ref_doc.get('content', '') if ref_doc else '')
                adjusted_random = adjust_writing_style_local(user_input, random_doc.get('content', '') if random_doc else '')
            except Exception as e:
                st.error(f"风格迁移错误: {e}")
                return

            st.subheader(">>> [方法A] 使用相似度最高的参考论文")
            st.write(f"相似度: {sim_score:.4f}")
            st.write(ref_doc.get('content', '')[:300] + "..." if ref_doc else "无参考文献")
            st.write(adjusted_similar)

            st.subheader(">>> [方法B] 使用随机选取的参考论文")
            st.write(random_doc.get('content', '')[:300] + "..." if random_doc else "无参考文献")
            st.write(adjusted_random)

        elif feature == "语义纠错":
            # 拼写纠错
            pinyin_info = get_pinyin_with_tone(user_input)
            spelling_prompt = (
                "你是中文拼写纠错专家，请根据拼音信息判断并纠正文本中可能的拼写错误。\n"
                f"文本：{user_input}\n拼音：{pinyin_info}"
            )
            spelling_result = call_local_qwen(spelling_prompt)

            if ENABLE_SELF_REFLECTION:
                reflection_prompt = (
                    "你是中文拼写纠错检查员，请检查纠错结果是否符合要求，遵循最小变化原则，避免引入新错误。\n"
                    f"原句: {user_input}\n初始纠错结果: {spelling_result}\n请输出最终正确的句子:"
                )
                spelling_result = call_local_qwen(reflection_prompt)

            syntax_report = generate_syntax_analysis(spelling_result)
            grammar_prompt = (
                "你是一个优秀的中文语病纠错模型，请纠正以下句子中的语法问题，遵循最小改动原则。\n"
                f"句子：{spelling_result}\n语法分析：\n{syntax_report}"
            )
            grammar_result = call_local_qwen(grammar_prompt)

            if ENABLE_SELF_REFLECTION:
                grammar_reflection_prompt = (
                    "你是语病检查员，请检查纠错结果是否符合要求，遵循最小变化原则，避免引入新错误。\n"
                    f"原句: {user_input}\n初始纠错结果: {grammar_result}\n请输出最终正确的句子:"
                )
                grammar_result = call_local_qwen(grammar_reflection_prompt)

            st.subheader("【语义纠错结果】")
            st.write(grammar_result)

if __name__ == "__main__":
    main()
