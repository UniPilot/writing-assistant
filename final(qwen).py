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
import hashlib

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
def hash_password(password: str) -> str:
    """密码哈希"""
    return hashlib.sha256(password.encode()).hexdigest()


def get_user_collection():
    """获取用户集合"""
    client = st.session_state.mongodb_client
    db = client.get_database("paper")
    return db.get_collection("users")


def register_user(username, password):
    """注册新用户"""
    users = get_user_collection()
    if users.find_one({"username": username}):
        return False, "用户名已存在"
    users.insert_one({"username": username, "password": hash_password(password)})
    return True, "注册成功"


def authenticate_user(username, password):
    """验证用户"""
    users = get_user_collection()
    user = users.find_one({"username": username})
    if user and user["password"] == hash_password(password):
        return True
    return False


def login_or_register():
    """登录 / 注册界面"""
    st.markdown("### 🔐 用户登录/注册")
    mode = st.radio("选择操作", ["登录", "注册"], horizontal=True)
    username = st.text_input("用户名")
    password = st.text_input("密码", type="password")

    if mode == "注册":
        if st.button("注册"):
            success, msg = register_user(username, password)
            if success:
                st.success(msg)
                st.session_state.username = username
                st.rerun()
            else:
                st.error(msg)
    else:  # 登录模式
        if st.button("登录"):
            if authenticate_user(username, password):
                st.session_state.username = username
                st.success(f"欢迎回来，{username}！")
                st.rerun()
            else:
                st.error("用户名或密码错误！")


def check_login():
    """检查登录状态"""
    if "username" not in st.session_state:
        login_or_register()
        st.stop()
def login_register():
    if 'users' not in st.session_state:
        st.session_state.users = {}
    st.sidebar.title("用户登录 / 注册")
    mode = st.sidebar.selectbox("选择操作", ["登录", "注册"])
    username = st.sidebar.text_input("用户名")
    password = st.sidebar.text_input("密码", type="password")
    if st.sidebar.button(mode):
        if not username or not password:
            st.sidebar.warning("用户名和密码不能为空")
            return False
        if mode == "注册":
            if username in st.session_state.users:
                st.sidebar.error("用户已存在")
            else:
                st.session_state.users[username] = hash_password(password)
                st.sidebar.success("注册成功，请登录")
        else:
            if username not in st.session_state.users:
                st.sidebar.error("用户不存在，请先注册")
            elif st.session_state.users[username] != hash_password(password):
                st.sidebar.error("密码错误")
            else:
                st.session_state.username = username
                st.sidebar.success(f"欢迎 {username}！")
                return True
    return 'username' in st.session_state

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
    with torch.no_grad(): outputs = model(**tokens)
    return outputs.last_hidden_state[:, 0, :]

def find_most_similar(input_text, collection, tokenizer, model):
    input_embedding = get_embedding(input_text, tokenizer, model)
    max_similarity = -1
    most_similar_article = None
    for doc in collection.find().limit(100):
        if 'content' not in doc or not doc['content']: continue
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
    for token in doc: lines.append(
        f"词语: {token.text:<5} 词性: {token.pos_:<5} 依存关系: {token.dep_:<10} 支配词: {token.head.text}")
    lines.append("\n[命名实体识别]")
    if not doc.ents:
        lines.append("未识别到命名实体")
    else:
        for ent in doc.ents: lines.append(
            f"实体: {ent.text:<15} 类型: {ent.label_:<10} 位置: ({ent.start_char}-{ent.end_char})")
    return "\n".join(lines)


def summarize_user_focus_area():
    user_texts = [m["input"] for m in st.session_state[history_key] if m["type"] in ["风格迁移", "语义纠错"]]
    if not user_texts: return "未获取到用户历史提问"
    prompt = ("请阅读以下用户的提问历史，总结出其关注的学术领域，直接输出1-2个简洁关键词：\n" + "\n".join(user_texts))
    interest_tags = call_local_qwen(prompt)
    st.session_state["interest_tags"] = interest_tags
    return interest_tags
def generate_paper_overview_from_history(chat_history):
    history_texts = [m["input"] for m in chat_history if m["type"] in ["风格迁移", "文本纠错"]]
    if not history_texts:
        return {
            "研究目的": "暂无",
            "相关工作": "暂无",
            "实验内容": "暂无",
            "结论": "暂无",
            "未来方向": "暂无"
        }
    overview_prompt = (
        "你是一名学术助手，接下来我会提供多段用户的写作内容（来自用户历史提问），请你帮我根据这些内容，提取出一篇完整论文应包括的五个部分："
        "研究目的、相关工作、实验内容、结论、未来方向。\n\n"
        "如果无法从文本中提取某一部分，请写“暂无”。请使用如下格式输出：\n\n"
        "研究目的：...\n相关工作：...\n实验内容：...\n结论：...\n未来方向：...\n\n"
        f"以下是用户历史内容：\n{chr(10).join(history_texts)}"
    )
    response = call_local_qwen(overview_prompt)
    sections = {
        "研究目的": "暂无",
        "相关工作": "暂无",
        "实验内容": "暂无",
        "结论": "暂无",
        "未来方向": "暂无"
    }
    for key in sections:
        if f"{key}：" in response:
            try:
                content = response.split(f"{key}：")[1].split("\n")[0].strip()
                if content:
                    sections[key] = content
            except:
                pass
    return sections

def generate_personalized_suggestions(focus_area, user_input_text):
    prompt = (
        f"已知用户关注的学科领域包括：{focus_area}。\n以下是用户在该领域内撰写的文本内容:\n{user_input_text}\n请结合用户关注的学术领域，根据该领域学术的写作规范，指出用户输入的文本中存在的一些问题，并从与用户输入中举出一些例子印证，最后再提出用户在文章结构、写作风格、逻辑表达或术语使用方面需要逐步提升的方向，并用'你'为称呼。")
    return call_local_qwen(prompt)

def main():
    st.set_page_config(page_title="学术写作智能助手", layout="wide")

    if not login_register():
        st.info("请先登录或注册")
        return

    username = st.session_state.username
    history_key = f"chat_history_{username}"
    if history_key not in st.session_state:
        st.session_state[history_key] = []

    st.markdown(f"<h1 style='text-align:center; color:#4A90E2;'>学术写作智能助手 - 用户：{username}</h1>", unsafe_allow_html=True)

    # 你的初始化代码
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
        enable_self_reflection = st.toggle("自我反思功能", value=True) if feature == "文本纠错" else False
        st.markdown("---")
        st.subheader("聊天历史")
        for i, chat in enumerate(st.session_state[history_key]):
            st.markdown(f"**{i + 1}. [{chat['type']}]** {chat['input'][:20]}...")

    for chat in st.session_state[history_key]:
        with st.chat_message("user"):
            st.markdown(chat["input"])
        output = chat.get("highlight_html") or chat.get("corrected_output") or chat.get("adjusted_output") or chat.get("suggestions") or ""
        with st.chat_message("assistant"):
            if chat.get("highlight_html"):
                st.markdown(chat["highlight_html"], unsafe_allow_html=True)
            else:
                st.markdown(output)

    if input_text := st.chat_input("请输入您要处理的文本..."):
        st.session_state[history_key].append({"type": feature, "input": input_text})

        with st.spinner("AI 正在处理，请稍候..."):
            if feature == "文本纠错":
                pinyin_info = get_pinyin_with_tone(input_text)
                spelling_prompt = (
                    f"你是中文拼写纠错专家，不需要判断文本内容是否合理，而是根据拼音信息，判断并纠正中文文本中可能存在的拼写错误,如果文本中有拼写错误，请直接输出修改后的句子，无需添加任何额外的解释或说明，如果输入的句子中不存在拼写错误，则直接输出原句即可。文本：{input_text}\n拼音：{pinyin_info}请直接输出最终正确的句子,不要给出其他多余文字:")
                spelling_result = call_local_qwen(spelling_prompt)

                if enable_self_reflection:
                    reflection_prompt = (
                        f"请检查以下纠错结果是否符合要求：\n1. 是否解决了原句中的所有拼写问题\n2. 是否遵循了最小变化原则\n3. 是否引入了新的错误\n4. 如果发现问题，请直接输出改进后的句子，无需解释;如果结果正确，请直接输出原句\n原句: {input_text}\n初始纠错结果: {spelling_result}\n\n请直接输出最终正确的句子,不要给出其他多余文字:")
                    spelling_result = call_local_qwen(reflection_prompt)

                if len(input_text) <= 150:
                    syntax_report = generate_syntax_analysis(spelling_result, st.session_state.spacy_model)
                    grammar_prompt = (
                        f"你是一个优秀的中文语病纠错模型，参考提供的句法分析报告，你需要识别并纠正输入的文本中可能含有的语病错误并输出正确的文本，纠正时尽可能减少对原文本的改动，并符合最小变化原则，即保证进行的修改都是最小且必要的，你应该避免对文章结构或词汇表达风格进行的修改。要求直接输出没有语法错误的句子，无需添加任何额外的解释或说明，如果输入的句子中不存在语法错误，则直接输出原句即可。句子：{spelling_result}\n语法分析结果：\n{syntax_report}请直接输出正确的文本,不要给出其他多余文字:")
                    grammar_result = call_local_qwen(grammar_prompt)
                else:
                    grammar_prompt = (
                        f"你是一个优秀的中文语病纠错模型，你需要识别并纠正输入的文本中可能含有的语病错误并输出正确的文本，纠正时尽可能减少对原文本的改动，并符合最小变化原则，即保证进行的修改都是最小且必要的，你应该避免对文章结构或词汇表达风格进行的修改。要求直接输出没有语法错误的句子，无需添加任何额外的解释或说明，如果输入的句子中不存在语法错误，则直接输出原句即可。句子：{spelling_result}\n请直接输出正确的文本,不要给出其他多余文字:")
                    grammar_result = call_local_qwen(grammar_prompt)

                if enable_self_reflection:
                    grammar_reflection_prompt = (
                        f"你是语病检查员，请检查以下纠错结果是否符合要求：\n1. 是否解决了原句中的所有语病问题\n2. 是否遵循了最小变化原则\n3. 是否引入了新的错误\n4. 如果发现问题，请直接输出改进后的句子，无需解释；如果用户初始纠错结果正确，请直接输出初始纠错结果,不需要说明\n原句: {input_text}\n初始纠错结果: {grammar_result}\n\n请直接输出最终正确的句子，不需要其他多余文字:")
                    grammar_result = call_local_qwen(grammar_reflection_prompt)

                # 将最终结果保存到 chat_history
                # 高亮版本
                highlight_html = diff_highlight(input_text, grammar_result)
                st.session_state[history_key][-1]["corrected_output"] = grammar_result
                st.session_state[history_key][-1]["highlight_html"] = highlight_html

                # 你的文本纠错处理逻辑，使用 st.session_state[history_key] 代替 chat_history
                ...
                # 例： st.session_state[history_key][-1]["corrected_output"] = grammar_result
            elif feature == "风格迁移":
                db = st.session_state.mongodb_client.get_database("paper")
                collection = db.get_collection("papers")
                ref_doc, sim_score = find_most_similar(input_text, collection, st.session_state.bert_tokenizer,
                                                       st.session_state.bert_model)

                if ref_doc:
                    adjusted_similar = adjust_writing_style_local(input_text, ref_doc.get('content', ''))
                    output_message = f"**根据风格相似度最高的三篇论文（最高相似度: {sim_score:.4f}）优化后：**\n\n---\n\n{adjusted_similar}"
                else:
                    output_message = "抱歉，数据库中未能找到相似的参考论文。请尝试其他文本或检查数据库。"
                # 将最终结果保存到 chat_history
                st.session_state[history_key][-1]["adjusted_output"] = output_message

                # 分支三：个性化建议

            elif feature == "个性化建议":
                if not any(chat['type'] in ["风格迁移", "文本纠错"] for chat in st.session_state[history_key]):
                    suggestions = "暂无历史记录，无法生成个性化建议。请先使用“文本纠错”或“风格迁移”功能。"
                else:
                    overview = generate_paper_overview_from_history(st.session_state[history_key])
                    overview_text = "\n".join([f"{k}：{v}" for k, v in overview.items()])
                    prompt = (
                        f"以下是用户当前撰写的文本：\n{input_text}\n\n"
                        f"以下是根据用户历史写作提取的论文概览信息：\n{overview_text}\n\n"
                        "请你结合用户当前文本与这些概览信息，指出其文本内容存在的主要问题，"
                        "并提供详细建议和修改方向。你可以引用概览内容作为参考来判断当前文本是否偏离原意或风格。请直接用“你”来称呼用户，格式清晰、条理明确。"
                    )
                    suggestions = call_local_qwen(prompt)
                # 将最终结果保存到 chat_history
                st.session_state[history_key][-1]["suggestions"] = suggestions
        # 处理完毕后刷新界面并避免无限循环，先标记再rerun
        st.session_state['just_updated'] = True

    if st.session_state.get('just_updated', False):
        st.session_state['just_updated'] = False
        st.rerun()

if __name__ == "__main__":
    main()
