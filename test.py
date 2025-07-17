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
    st.title("学术写作智能助手")

    # 初始化历史记录
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 连接 MongoDB 和 BERT 模型
    if "mongodb_client" not in st.session_state:
        st.session_state.mongodb_client = connect_mongodb()
    if "bert_tokenizer" not in st.session_state or "bert_model" not in st.session_state:
        tokenizer, model = initialize_bert()
        st.session_state.bert_tokenizer = tokenizer
        st.session_state.bert_model = model

    feature = st.radio("请选择功能", ["风格迁移", "语义纠错"], horizontal=True)
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

            ref_doc, sim_score = find_most_similar(user_input, collection, st.session_state.bert_tokenizer, st.session_state.bert_model)
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

            # 添加风格迁移历史记录
            st.session_state.chat_history.append({
                "type": "风格迁移",
                "input": user_input,
                "reference_excerpt": ref_doc.get('content', '')[:200] if ref_doc else "",
                "adjusted_output": adjusted_similar
            })

        elif feature == "语义纠错":
            pinyin_info = get_pinyin_with_tone(user_input)
            spelling_prompt = (
                "你是中文拼写纠错专家，请根据拼音信息，判断并纠正中文文本中可能存在的拼写错误。\n"
                "要求：\n1. 只输出纠正后的文本\n2. 若文本无误，则原样输出\n3. 避免引入新错误\n"
                f"文本：{user_input}\n拼音：{pinyin_info}"
            )
            spelling_result = call_local_qwen(spelling_prompt)

            if ENABLE_SELF_REFLECTION:
                reflection_prompt = (
                    f"请检查以下纠错结果是否符合要求：\n"
                    f"1. 是否解决了原句中的所有拼写问题\n"
                    f"2. 是否遵循了最小变化原则\n"
                    f"3. 是否引入了新的错误\n"
                    f"4. 如果发现问题，请直接输出改进后的句子，无需解释\n"
                    f"5. 如果结果正确，请直接输出原句\n\n"
                    f"原句: {user_input}\n"
                    f"初始纠错结果: {spelling_result}\n\n"
                    f"请输出最终正确的句子:"
                )
                spelling_result = call_local_qwen(reflection_prompt)

            syntax_report = generate_syntax_analysis(spelling_result)
            grammar_prompt = (
                "你是一个优秀的中文语病纠错模型，你需要识别并纠正输入的句子中可能含有的语病错误并输出正确的句子，参考提供的句法分析报告，纠正时尽可能减少对原句子的改动，并符合最小变化原则，即保证进行的修改都是最小且必要的。你应该避免对句子结构或词汇表达进行不必要的修改。要求直接输出没有语法错误的句子，无需添加任何额外的解释或说明，如果输入的句子中不存在语法错误，则直接输出原句即可。"
                f"句子：{spelling_result}\n语法分析：\n{syntax_report}"
            )
            grammar_result = call_local_qwen(grammar_prompt)

            if ENABLE_SELF_REFLECTION:
                grammar_reflection_prompt = (
                    f"你是语病检查员，请检查以下纠错结果是否符合要求：\n"
                    f"1. 是否解决了原句中的所有语病问题\n"
                    f"2. 是否遵循了最小变化原则\n"
                    f"3. 是否引入了新的错误\n"
                    f"4. 如果发现问题，请直接输出改进后的句子，无需解释\n"
                    f"5. 如果结果正确，请直接输出原句\n\n"
                    f"原句: {user_input}\n"
                    f"初始纠错结果: {grammar_result}\n\n"
                    f"请输出最终正确的句子:"
                )
                grammar_result = call_local_qwen(grammar_reflection_prompt)

            st.subheader("【语义纠错结果】")
            st.write(grammar_result)

            # 添加语义纠错历史记录
            st.session_state.chat_history.append({
                "type": "语义纠错",
                "input": user_input,
                "corrected_output": grammar_result
            })

    # 显示历史记录
    with st.expander("历史对话记录", expanded=False):
        if not st.session_state.chat_history:
            st.write("暂无历史记录。")
        else:
            for i, record in enumerate(st.session_state.chat_history[::-1], 1):
                st.markdown(f"**记录 {i}**")
                st.markdown(f"**类型：** {record['type']}")
                st.markdown(f"**原始输入：** {record['input']}")
                if record["type"] == "风格迁移":
                    st.markdown(f"**参考片段：** {record['reference_excerpt']}...")
                    st.markdown(f"**生成结果：** {record['adjusted_output']}")
                else:
                    st.markdown(f"**纠错结果：** {record['corrected_output']}")
                st.markdown("---")

if __name__ == "__main__":
    main()
