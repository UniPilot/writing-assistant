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
    uri = (
        "mongodb://2068432802:lzq520796@"
        "cluster0-shard-00-00.tmy62.mongodb.net:27017,"
        "cluster0-shard-00-01.tmy62.mongodb.net:27017,"
        "cluster0-shard-00-02.tmy62.mongodb.net:27017/"
        "?ssl=true&replicaSet=atlas-j16zyw-shard-0&authSource=admin"
        "&retryWrites=true&w=majority&appName=Cluster0"
    )

    client = MongoClient(uri,
                         tlsCAFile=certifi.where(),
                         server_api=ServerApi('1'))
    try:
        client.admin.command('ping')
        print("成功连接到 MongoDB！")
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
        "不要增加原文没有的内容，不要根据范文内容进行修改，而只修改写作风格，但修改用词、句式和结构以尽量匹配参考论文的学术风格。\n"
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

def summarize_user_focus_area():
    # 从历史记录中提取用户输入
    user_texts = [m["input"] for m in st.session_state.chat_history if m["type"] in ["风格迁移", "语义纠错"]]
    if not user_texts:
        return "未获取到用户历史提问"
    prompt = (
        "请阅读以下用户的提问历史，总结出其关注的学术领域，直接输出1-2个简洁关键词：\n"
        + "\n".join(user_texts)
    )
    interest_tags = call_local_qwen(prompt)
    st.session_state["interest_tags"] = interest_tags  # 缓存
    return interest_tags

def generate_personalized_suggestions(focus_area,user_input_text):
    prompt = (
        f"已知用户关注的学科领域包括：{focus_area}。\n"
        f"以下是用户在该领域内撰写的文本内容:{user_input_text}\n"
        f"请结合用户关注的学术领域，根据该领域学术的写作规范，指出用户输入的文本中存在的一些问题，并从与用户输入中举出一些例子印证，最后再提出用户在文章结构、写作风格、逻辑表达或术语使用方面需要逐步提升的方向，并用'你'为称呼"
    )
    return call_local_qwen(prompt)


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

    feature = st.radio("请选择功能", ["风格迁移", "语义纠错","个性化建议"], horizontal=True)
    user_input = st.text_area(f"请输入文本（当前功能：{feature}）", height=150)

    if st.button("执行"):
        if feature != "个性化建议" and not user_input.strip():
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

            st.write(f"相似度: {sim_score:.4f}")
            st.write(ref_doc.get('content', '')[:300] + "..." if ref_doc else "无参考文献")
            st.write(adjusted_similar)

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
                "你是中文拼写纠错专家，不需要判断文本内容是否合理，而是根据拼音信息，判断并纠正中文文本中可能存在的拼写错误,如果文本中有拼写错误，请直接输出修改后的句子，无需添加任何额外的解释或说明，如果输入的句子中不存在拼写错误，则直接输出原句即可。"
                f"文本：{user_input}\n拼音：{pinyin_info}"
                f"请直接输出最终正确的句子,不要给出其他多余文字:"
            )
            spelling_result = call_local_qwen(spelling_prompt)
            if ENABLE_SELF_REFLECTION:
                reflection_prompt = (
                    f"请检查以下纠错结果是否符合要求：\n"
                    f"1. 是否解决了原句中的所有拼写问题\n"
                    f"2. 是否遵循了最小变化原则\n"
                    f"3. 是否引入了新的错误\n"
                    f"4. 如果发现问题，请直接输出改进后的句子，无需解释;如果结果正确，请直接输出原句\n"
                    f"原句: {user_input}\n"
                    f"初始纠错结果: {spelling_result}\n\n"
                    f"请直接输出最终正确的句子,不要给出其他多余文字:"
                )
                spelling_result = call_local_qwen(reflection_prompt)
            if len(user_input) <= 150:
                syntax_report = generate_syntax_analysis(spelling_result)
                grammar_prompt = (
                    f"你是一个优秀的中文语病纠错模型，参考提供的句法分析报告，你需要识别并纠正输入的文本中可能含有的语病错误并输出正确的文本，纠正时尽可能减少对原文本的改动，并符合最小变化原则，即保证进行的修改都是最小且必要的，你应该避免对文章结构或词汇表达风格进行的修改。要求直接输出没有语法错误的句子，无需添加任何额外的解释或说明，如果输入的句子中不存在语法错误，则直接输出原句即可。"
                    f"句子：{spelling_result}\n语法分析结果：\n{syntax_report}"
                    f"请直接输出正确的文本,不要给出其他多余文字:"
                )
                grammar_result = call_local_qwen(grammar_prompt)
            else:
                grammar_prompt = (
                    f"你是一个优秀的中文语病纠错模型，你需要识别并纠正输入的文本中可能含有的语病错误并输出正确的文本，纠正时尽可能减少对原文本的改动，并符合最小变化原则，即保证进行的修改都是最小且必要的，你应该避免对文章结构或词汇表达风格进行的修改。要求直接输出没有语法错误的句子，无需添加任何额外的解释或说明，如果输入的句子中不存在语法错误，则直接输出原句即可。"
                    f"句子：{spelling_result}\n"
                    f"请直接输出正确的文本,不要给出其他多余文字:"
                )
                grammar_result = call_local_qwen(grammar_prompt)
            if ENABLE_SELF_REFLECTION:
                grammar_reflection_prompt = (
                    f"你是语病检查员，请检查以下纠错结果是否符合要求：\n"
                    f"1. 是否解决了原句中的所有语病问题\n"
                    f"2. 是否遵循了最小变化原则\n"
                    f"3. 是否引入了新的错误\n"
                    f"4. 如果发现问题，请直接输出改进后的句子，无需解释；如果用户初始纠错结果正确，请直接输出初始纠错结果,不需要说明\n"
                    f"原句: {user_input}\n"
                    f"初始纠错结果: {grammar_result}\n\n"
                    f"请直接输出最终正确的句子，不需要其他多余文字:"
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


        elif feature == "个性化建议":
            if not st.session_state.chat_history:
                st.warning("暂无历史记录，无法生成个性化建议，请先使用风格迁移或语义纠错功能。")
                return
            st.write("正在分析用户研究方向和写作问题，请稍候...")
            # 1. 获取用户关注领域
            focus_area = summarize_user_focus_area()
            # 2. 提取最近的写作输出用于分析
            recent_inputs = [
                record["input"]
                for record in st.session_state.chat_history
                if record["type"] in ["风格迁移", "语义纠错"]
            ]
            if not recent_inputs:
                st.warning("历史记录中无可用于分析的文本结果。")
                return
            # 5. 生成个性化建议
            input_text = recent_inputs
            suggestions = generate_personalized_suggestions(focus_area,input_text)
            st.subheader("【个性化写作建议】")
            st.write(suggestions)
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
                elif record["type"] == "语义纠错":
                    st.markdown(f"**纠错结果：** {record['corrected_output']}")
                elif record["type"] == "个性化建议":
                    st.markdown(f"**用户领域：** {record['focus_area']}")
                    st.markdown(f"**识别问题：** {record['problems']}")
                    st.markdown(f"**个性化建议：** {record['suggestions']}")
                st.markdown("---")

if __name__ == "__main__":
    main()
