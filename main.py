from reminder import reminder_manager
from datetime import datetime
import os
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
import streamlit as st
import requests
import json
import time
import os
import mysql.connector
from pypinyin import lazy_pinyin, Style
import sys
import pandas as pd
import threading
import matplotlib.pyplot as plt
import seaborn as sns
import re

# 添加模块路径（根据项目结构调整）
sys.path.append('.')
sys.path.append('..')

# ✅ 修改：使用学校服务器 API 配置
api_key = "not empty"  # 根据实际情况可能需要填写真实 API key
base_url = "http://10.0.30.172:9997/v1"  # 学校服务器 API 地址
model_name = "qwen2.5-vl-instruct"  # 使用学校服务器上的模型

from rag_answer import get_rag_answer_or_fallback, is_health_related  # 新增导入is_health_related
from utils.utils import name_to_pinyin_abbr, ensure_dir
from utils.last_conversation import get_latest_conversation_path
from health.health_extractor import extract_health_from_latest_conversation
from mood.mood_handler import handle_mood_and_greeting
from database.connect_sql import (
    get_user_id_by_name,
    update_user_health,
    insert_new_user,
    get_user_profile_by_name,
    update_user_emotional_needs
)
from emotion.emotion_extractor import EmotionNeedsExtractor, EMOTION_DICT
from health.health_logger import (
    analyze_health_log_from_conversation,
    save_health_log_to_db,
    display_user_health_logs_with_timestamp
)
from mood.portemotion import analyze_sentence_and_image
from emotion.emotion_log import log_emotional_need, display_emotional_need_timeline
from utils.conversation_history_manage import get_latest_three_conversations

# 页面配置
st.set_page_config(
    page_title="老年对话助手",  # 修改标题
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ====== 全局样式配置 (统一风格) ======
# 将第一个页面的 CSS 提取为全局样式，应用到所有页面
GLOBAL_STYLES = """
<style>
/* 全局背景色 */
.stApp {
    background-color: #E8F4FF;
}

/* 顶部装饰条 */
.top-bar {
    height: 8px;
    background: linear-gradient(90deg, #4A90E2, #D6EAF8);
    width: 100%;
    position: fixed;
    top: 0;
    left: 0;
    z-index: 999;
}

/* 标题样式 */
h1 {
    color: #4A90E2 !important;
    font-weight: 700 !important;
    font-size: 2.2rem !important;
    text-align: center;
    padding-bottom: 25px;
    font-family: 'Segoe UI', sans-serif;
}
h2, h3 {
    color: #4A90E2 !important;
    font-family: 'Segoe UI', sans-serif;
}

/* 滚动容器/Expander/卡片背景 样式 */
div[data-testid="stVerticalBlockBorderWrapper"], .streamlit-expanderContent {
    background-color: rgba(255, 255, 255, 0.6) !important;
    border: 1px solid white;
    border-radius: 16px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.02);
}

/* 通用按钮样式 (次级/默认) - 卡片化风格 */
div.stButton > button[kind="secondary"] {
    background-color: white;
    color: #333;
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 15px 20px;
    font-size: 16px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    transition: all 0.25s ease;
    height: auto;
    min-height: 60px;
}

/* 悬停效果 */
div.stButton > button[kind="secondary"]:hover {
    background-color: #D6EAF8;
    border-color: #4A90E2;
    box-shadow: 0 6px 12px rgba(74, 144, 226, 0.15);
    color: #0056b3;
    transform: translateY(-2px);
}
div.stButton > button[kind="secondary"]:active {
    background-color: #badcf5;
    transform: translateY(0px);
}

/* 主要按钮样式 (Primary) */
div.stButton > button[kind="primary"] {
    background-color: #4A90E2;
    color: white;
    border: none;
    border-radius: 30px;
    padding: 12px 30px;
    font-size: 1.1rem;
    font-weight: 600;
    box-shadow: 0 4px 10px rgba(74, 144, 226, 0.3);
}
div.stButton > button[kind="primary"]:hover {
    background-color: #357ABD;
    box-shadow: 0 6px 15px rgba(74, 144, 226, 0.4);
    transform: scale(1.02);
}

/* 聊天框修正 */
.stChatMessage {
    background-color: rgba(255, 255, 255, 0.5);
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
}
.rag-answer {
    background-color: #f0f8ff;
    border-left: 4px solid #4a90e2;
    padding: 15px;
    margin: 10px 0;
    border-radius: 8px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}
</style>
<div class="top-bar"></div>
"""
st.markdown(GLOBAL_STYLES, unsafe_allow_html=True)


# 数据库连接配置
DB_CONFIG = {
    "host": "10.0.30.172",
    "port": 13306,
    "user": "root",
    "password": "123456",
    "database": "talk"
}

# 系统提示词
SYSTEM_PROMPT = """
你是一个温暖、耐心且富有同理心的对话助手，专门为陪伴老年人而设计。你的名字叫小新，目标是提供情感支持、倾听他们的故事，并鼓励他们表达情绪。
    请遵循以下原则：
    1. 在表达情绪时要适当加入一些黄豆表情，来增加亲切感。
    2. 回答一定要简短易懂，不要太长！语气温和亲切，就像朋友聊天一样。
    3. 当老人感到孤独或焦虑时，请给予安慰和理解。
    4. 不要老是问问题。
    5.特别注意：当回答健康医疗相关问题时，务必提供完整的信息，包括可能的症状、建议和注意事项。
    6.回答的时候语句要简短，不要回答一大串话。
    请记住，你的角色不仅是回答问题，更是陪伴、倾听和关心他们的心情。
"""

# RAG 配置
RAG_THRESHOLD = 0.5  # 相似度阈值，可调整

# 初始化情感需求提取器（全局单例实例）
emotion_extractor = EmotionNeedsExtractor()

# 页面状态初始化
def init_session_state():
    if 'page' not in st.session_state:
        st.session_state.page = "select_user"
    if 'selected_user' not in st.session_state:
        st.session_state.selected_user = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.7
    if 'top_p' not in st.session_state:
        st.session_state.top_p = 0.8
    if 'max_tokens' not in st.session_state:
        st.session_state.max_tokens = 2048
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    if "last_response_time" not in st.session_state:
        st.session_state.last_response_time = "无"
    if "selected_conversation" not in st.session_state:
        st.session_state.selected_conversation = None
    if "emotion" not in st.session_state:
        st.session_state.emotion = None
    # 新增：存储情感需求历史
    if "emotional_needs_history" not in st.session_state:
        st.session_state.emotional_needs_history = []
    # 新增：记录对话开始时间
    if "conversation_start_time" not in st.session_state:
        st.session_state.conversation_start_time = None
    # 新增：RAG 相关状态
    if "rag_enabled" not in st.session_state:
        st.session_state.rag_enabled = True
    if "rag_threshold" not in st.session_state:
        st.session_state.rag_threshold = RAG_THRESHOLD
 
init_session_state()

def start_services() -> None:
    if not reminder_manager.running:
        try:
            reminder_manager.start()
        except Exception:
            pass

# 调用启动函数
start_services()

def extract_recent_health_issues(conversations: list) -> list:
    """提取最近对话中提到的健康问题"""
    if not conversations:
        return []
    
    # 健康相关关键词
    health_keywords = ['感冒', '发烧', '咳嗽', '头疼', '头晕', '高血压', '糖尿病', '帕金森', '阿尔茨海默', '失眠', '心脏病']
    
    # 只看用户最近5条消息
    user_messages = [msg['content'] for msg in conversations if msg['role'] == 'user'][-5:]
    
    mentioned_issues = []
    for msg in user_messages:
        for keyword in health_keywords:
            if keyword in msg and keyword not in mentioned_issues:
                mentioned_issues.append(keyword)
    
    return mentioned_issues

def generate_history_reminder(health_issues: list) -> str:
    """生成历史健康问题的提醒文本"""
    if not health_issues:
        return ""
    
    reminder = "\n\n历史对话提醒：\n"
    for issue in health_issues:
        reminder += f"- 用户之前提到过{issue}，请在合适的时机关心其恢复情况\n"
    
    return reminder
# 获取用户列表
def get_users():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM users")
        result = [row[0] for row in cursor.fetchall()]
        conn.close()
        return result
    except Exception as e:
        st.error(f"数据库连接失败: {e}")
        return []

# 创建新对话函数
def create_new_conversation():
    if st.session_state.messages and st.session_state.selected_user:
        # 对话结束时间（点击"新建对话"的时间）
        conversation_end_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # 对话开始时间（用户首次发消息的时间）
        conversation_start_time = st.session_state.conversation_start_time

        new_conversation = {
            "messages": st.session_state.messages.copy(),
            "start_time": conversation_start_time,
            "end_time": conversation_end_time,
            "conversation_id": st.session_state.conversation_id
        }
        st.session_state.conversation_history.append(new_conversation)

        selected_user = st.session_state.selected_user
        user_id = get_user_id_by_name(selected_user)
        if user_id is None:
            st.error("无法获取用户 ID，无法保存对话历史")
            return

        # 提取当前对话的情感需求
        from emotion.emotion_extractor import EmotionNeedsExtractor
        from emotion.emotion_log import log_emotional_need

        extractor = EmotionNeedsExtractor()
        all_emotional_needs = []

        for message in st.session_state.messages:
            if message["role"] == "user":
                needs = extractor.extract_needs(message["content"])
                all_emotional_needs.extend(needs)

        # 去重后更新 profiles 表中的情感需求字段
        unique_needs = list(set(all_emotional_needs))
        update_result = update_user_emotional_needs(user_id, unique_needs)

        # 记录每条情感需求到日志表（带对话结束时间戳）
        if unique_needs:
            log_emotional_need(user_id, all_emotional_needs, conversation_end_time)

        # 新增：提取情感需求并拼接到每条用户消息后
        from emotion.emotion_extractor import EmotionNeedsExtractor
        extractor = EmotionNeedsExtractor()

        # 构建新的消息列表
        new_messages = []
        for message in st.session_state.messages:
            if message["role"] == "user":
                needs = extractor.extract_needs(message["content"])
                content_with_emotion = f"{message['content']}（情感需求：{', '.join(needs)}）"
                new_messages.append({
                    "role": "user",
                    "content": content_with_emotion
                })
            else:
                new_messages.append(message.copy())

        # 写入文件
        pinyin = name_to_pinyin_abbr(selected_user)
        folder_name = f"{pinyin}_{user_id}"
        history_dir = os.path.join('history', folder_name)
        ensure_dir(history_dir)
        file_name = os.path.join(history_dir, f'conversation_{conversation_end_time}.txt')
        with open(file_name, 'w', encoding='utf-8') as f:
            for message in new_messages:
                f.write(f"{message['role']}: {message['content']}\n")

        # 保留原有逻辑：提取并更新健康信息
        try:
            latest_file = get_latest_conversation_path(folder_name)  # ❌ 虽然可能正确，但不保险
            health_keywords = extract_health_from_latest_conversation(latest_file)
            health_str = ', '.join(health_keywords)
            update_user_health(user_id, health_str)
            health_logs = analyze_health_log_from_conversation(latest_file)
            save_health_log_to_db(user_id, health_logs)
            st.success("✅ 成功更新用户健康信息（动态+日志）")
        except Exception as e:
            st.warning(f"⚠️ 健康信息更新失败: {str(e)}")

    # 清空当前对话状态
    st.session_state.messages = []
    st.session_state.conversation_id = None
    st.session_state.selected_conversation = None
    st.session_state.conversation_start_time = None


# ████████████████ 用户选择页 ████████████████ #
if st.session_state.page == "select_user":
    # 标题区域
    st.markdown("<h1>👋 请选择要对话的用户</h1>", unsafe_allow_html=True)
    
    users = get_users()
    if not users:
        st.warning("没有可用用户，请检查数据库是否正常。")
    else:
        # 滚动容器
        with st.container(height=500, border=True):
            for name in users:
                # 增加空格调整图标间距
                if st.button(f"👤   {name}", key=f"user_{name}", use_container_width=True):
                    st.session_state.selected_user = name
                    st.session_state.page = "dashboard"
                    st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # 底部创建用户按钮，居中显示
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button("➕ 创建新用户", type="primary", use_container_width=True):
            st.session_state.page = "create_user"
            st.rerun()

    st.markdown("---")

    st.stop()

# ████████████████ 用户主页仪表盘（新增） ████████████████ #
elif st.session_state.page == "dashboard":

    st.markdown(f"<h1>🧓 欢迎，{st.session_state.selected_user}</h1>", unsafe_allow_html=True)
    st.write("请选择功能：")

    st.divider()

    # 创建 3 列宫格布局
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("💬 开始聊天", key="chat_btn", use_container_width=True):
            st.session_state.page = "chat"
            st.rerun()
        if st.button("😊 情绪识别", key="mood_btn", use_container_width=True):
            st.session_state.page = "detect_mood"
            st.rerun()

    with col2:
        if st.button("📅 健康日志", key="health_log_btn", use_container_width=True):
            st.session_state.page = "health_log"
            st.rerun()
        if st.button("📊 健康可视化", key="health_visual_btn", use_container_width=True):
            st.session_state.page = "health_visualization"
            st.rerun()

    with col3:
        if st.button("❤️ 情感需求统计", key="emotion_stat_btn", use_container_width=True):
            st.session_state.page = "emotion_need_stats"
            st.rerun()
        if st.button("📈 情感可视化", key="emotion_visual_btn", use_container_width=True):
            st.session_state.page = "emotion_visualization"
            st.rerun()

    st.divider()

    st.markdown("### 🔧 更多功能")

    # 历史记录按钮
    if st.button("📝 查看历史对话记录", use_container_width=True):
        st.session_state.page = "conversation_history"
        st.rerun()
    # ⭐ 新增：查看对话总结
    if st.button("🧾 查看对话总结", use_container_width=True):
        st.session_state.page = "conversation_summary"
        st.rerun()
    # ⭐ 新增：提醒事项查看按钮
    if st.button("⏰ 查看提醒事项", use_container_width=True):
        st.session_state.page = "reminder_view"
        st.rerun()

    # 返回用户选择页按钮
    if st.button("⬅️ 返回用户选择页", use_container_width=True):
        st.session_state.selected_user = None
        st.session_state.page = "select_user"
        st.rerun()

    st.stop()




# ████████████████ 情绪识别选择页（新增） ████████████████ #
elif st.session_state.page == "mood_choice":
    st.markdown("<h1>😊 是否需要情绪识别？</h1>", unsafe_allow_html=True)

    st.write("您可以选择是否先进行情绪识别。")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("开始情绪识别 🔍"):
            st.session_state.page = "detect_mood"   ### ← 进入原本的识别页面
            st.rerun()

    with col2:
        if st.button("跳过，直接开始对话 💬"):
            st.session_state.page = "chat"          ### ← 不识别直接进入聊天
            st.rerun()

    st.stop()
    
# ████████████████ 提醒事项查看页（新增） ████████████████ #
elif st.session_state.page == "reminder_view":
    st.markdown(f"<h1>⏰ {st.session_state.selected_user} 的提醒事项</h1>", unsafe_allow_html=True)

    from database.reminder_file import load_user_reminders

    user_id = get_user_id_by_name(st.session_state.selected_user)
    reminders = load_user_reminders(user_id)

    if not reminders:
        st.info("暂无提醒事项")
    else:
        st.subheader("📋 当前提醒事项")

        for rem in reminders:

            content = rem.get("content", "无内容")
            time_str = rem.get("time", "未知")
            rtype = rem.get("repeat_type", "none")
            created_at = rem.get("created_at", "")

            date = rem.get("date")
            weekdays = rem.get("weekdays")

            type_display = {
                "daily": "每日重复",
                "once": "单次提醒",
                "weekly": "每周重复"
            }.get(rtype, rtype)

            with st.container():
                st.markdown("---")  # 卡片分隔线

                st.markdown(f"### 📝 提醒内容：{content}")
                st.markdown(f"**⏰ 时间：** {time_str}")
                st.markdown(f"**🔁 类型：** {type_display}")

                if date:
                    st.markdown(f"**📅 日期：** {date}")

                if weekdays and rtype == "weekly":
                    st.markdown(f"**📆 每周：** {', '.join(map(str, weekdays))}")

                st.caption(f"🕒 创建时间：{created_at}")

    st.markdown("---")

    if st.button("🔙 返回主页", use_container_width=True):
        st.session_state.page = "dashboard"
        st.rerun()

    st.stop()




# ████████████████ 情绪识别页 ████████████████ #
elif st.session_state.page == "detect_mood":
    st.markdown("<h1>😊 情绪识别中...</h1>", unsafe_allow_html=True)
    st.write("我们正在通过面部/语音识别分析您当前的情绪状态，请稍等片刻...")
    # result = analyze_sentence_and_image("今天阳光明媚，我感觉很好", "happy.jpg", rmssd=35)
    # predict_emotion = result["predicted_emotion"]
    predict_emotion = "开心"
    predict_emotion, greeting = handle_mood_and_greeting(predict_emotion)
    st.session_state.emotion = predict_emotion
    st.write(f"🔍 识别到您的情绪为：**{predict_emotion}**")
    if st.button("开始对话", type="primary"):
        st.session_state.messages.append({"role": "assistant", "content": greeting})
        st.session_state.page = "chat"
        st.rerun()
    st.stop()

# ████████████████ 创建用户页 ████████████████ #
elif st.session_state.page == "create_user":
    st.markdown("<h1>创建新用户</h1>", unsafe_allow_html=True)
    new_user_name = st.text_input("请输入新用户名:")
    if st.button("创建用户", type="primary"):

        if not new_user_name:
            st.error("用户名不能为空！")
        else:
            try:
                user_id = insert_new_user(new_user_name)
                if user_id:
                    st.success(f"用户 '{new_user_name}' 创建成功！")
                    st.session_state.selected_user = new_user_name
                    st.session_state.page = "detect_mood"
                    st.rerun()
                else:
                    st.error("用户创建失败，请稍后再试。")
            except Exception as e:
                st.error(f"发生错误: {str(e)}")
    # ⭐ 新增：返回用户选择页按钮
    if st.button("⬅️ 返回用户选择页", use_container_width=True):
        st.session_state.page = "select_user"
        st.rerun()
    st.stop()

# ████████████████ 健康日志页 ████████████████ #
elif st.session_state.page == "health_log":
    st.markdown(f"<h1>📅 {st.session_state.selected_user} 的健康日志</h1>", unsafe_allow_html=True)
    user_id = get_user_id_by_name(st.session_state.selected_user)
    if user_id:
        display_user_health_logs_with_timestamp(user_id, use_streamlit=True)
    else:
        st.error("无法获取用户 ID，请确保数据库连接正常。")
    if st.button("🔙 返回主页", use_container_width=True):
        st.session_state.page = "dashboard"
        st.rerun()
    st.stop()
# ████████████████ 健康日志可视化页 ████████████████ #
elif st.session_state.page == "health_visualization":
    from health.health_visualization import plot_health_visualization  # 动态导入，避免循环依赖

    selected_user = st.session_state.selected_user
    if not selected_user:
        st.error("❌ 未选择用户，请返回对话页选择用户")
        if st.button("🔙 返回主页"):
            st.session_state.page = "dashboard"
            st.rerun()
        st.stop()

    user_id = get_user_id_by_name(selected_user)
    if not user_id:
        st.error("❌ 用户不存在，请检查数据库")
        if st.button("🔙 返回主页"):
            st.session_state.page = "dashboard"
            st.rerun()
        st.stop()

    st.markdown(f"<h1>📊 {selected_user} 的健康日志可视化</h1>", unsafe_allow_html=True)
    plot_health_visualization(user_id)  # 调用可视化函数

    if st.button("🔙 返回主页", use_container_width=True):
        st.session_state.page = "dashboard"
        st.rerun()

    st.stop()
# ████████████████ 情感需求统计页 ████████████████ #
elif st.session_state.page == "emotion_need_stats":
    st.markdown("<h1>📊 情感需求统计</h1>", unsafe_allow_html=True)

    selected_user = st.session_state.selected_user
    if not selected_user:
        st.error("❌ 未选择用户，请返回主页选择用户。")
        st.stop()

    user_id = get_user_id_by_name(selected_user)
    if not user_id:
        st.error("❌ 用户不存在，请检查数据库。")
        st.stop()

    # 调用 emotion_log.py 中的接口显示情感需求统计
    display_emotional_need_timeline(user_id)
    if st.button("🔙 返回主页", use_container_width=True):
        st.session_state.page = "dashboard"
        st.rerun()
    st.stop()
# ████████████████ 情感需求可视化页 ████████████████ #
elif st.session_state.page == "emotion_visualization":
    from emotion.emotion_view import plot_emotional_visualization  # 动态导入，避免循环依赖

    selected_user = st.session_state.selected_user
    if not selected_user:
        st.error("❌ 未选择用户，请返回对话页选择用户")
        if st.button("🔙 返回主页"):
            st.session_state.page = "dashboard"
            st.rerun()
        st.stop()

    user_id = get_user_id_by_name(selected_user)
    if not user_id:
        st.error("❌ 用户不存在，请检查数据库")
        if st.button("🔙 返回主页"):
            st.session_state.page = "dashboard"
            st.rerun()
        st.stop()

    st.markdown(f"<h1>📊 {selected_user} 的情感需求可视化</h1>", unsafe_allow_html=True)
    plot_emotional_visualization(user_id)  # 调用可视化函数

    if st.button("🔙 返回主页", use_container_width=True):
        st.session_state.page = "dashboard"
        st.rerun()
    st.stop()
######################################################################

# ████████████████ 对话历史查看页（新增） ████████████████ #
elif st.session_state.page == "conversation_history":
    
    st.markdown(f"<h1>📜 {st.session_state.selected_user} 的对话历史</h1>", unsafe_allow_html=True)

    convs = get_latest_three_conversations(st.session_state.selected_user)

    if not convs:
        st.info("该用户暂无历史对话。")
    else:
        for i, conv in enumerate(convs):
            st.subheader(f"对话 {i+1}：{conv.get('start_time')}")
            with st.expander("查看对话内容"):
                for msg in conv["messages"]:
                    role = "👤 用户" if msg["role"] == "user" else "🤖 助手"
                    st.write(f"**{role}:** {msg['content']}")

    if st.button("🔙 返回用户主页", use_container_width=True):
        st.session_state.page = "dashboard"
        st.rerun()

    st.stop()


# ████████████████ 对话总结展示页 ████████████████ #
elif st.session_state.page == "conversation_summary":
    # 注意：这里需要判断是哪个入口进入的
    elder_name = st.session_state.get("selected_elder") or st.session_state.get("selected_user")
    if not elder_name:
        st.error("未选择老人，请返回选择")
        if st.button("返回"):
            st.session_state.page = "select_user"
            st.rerun()
        st.stop()
    
    st.markdown(f"<h1>📝 {elder_name} 的对话总结</h1>", unsafe_allow_html=True)
    
    # 加载对话记录（只取最近3轮）
    with st.spinner("正在加载对话记录..."):
        from utils.conversation_history_manage import get_elder_conversations
        all_conversations = get_elder_conversations(elder_name)  # 获取所有对话（已按时间倒序）
        # 只取最近3轮用于总结和显示
        conversations = all_conversations[:3] if len(all_conversations) >= 3 else all_conversations
        conversation_count = len(conversations)
    
    if not conversations:
        st.info("该老人暂无对话记录")
    else:
        # 显示总结的对话轮数
        if conversation_count < 3:
            st.info(f"📌 该老人共有 {conversation_count} 轮对话，将总结全部对话")
        else:
            st.info(f"📌 总结最近3轮对话（共 {len(all_conversations)} 轮）")
        
        # 生成总结
        with st.spinner("正在生成总结..."):
            from rag_answer import summarize_conversations
            summary = summarize_conversations(conversations, elder_name)
        
        # 展示总结
        st.subheader("📋 对话总结")
        st.markdown(summary)
        
        # 展示原始对话记录（只显示用于总结的对话）
        st.markdown("---")
        st.subheader(f"📝 原始对话记录（共 {conversation_count} 轮）")
        for i, conv in enumerate(conversations, 1):
            with st.expander(f"第 {i} 轮对话（{'最新' if i == 1 else '较早'}）", expanded=(i == 1)):
                st.text_area(
                    f"对话内容 {i}",
                    conv,
                    height=200,
                    key=f"conversation_{i}",
                    label_visibility="collapsed"
                )
    
    # 返回按钮
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔙 返回主页面", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()

######################################################################
# # ████████████████ 聊天页 ████████████████ #
else:
    import json, os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BUFFER_FILE = os.path.join(BASE_DIR, "reminder_buffer.json")
    # ================ WebSocket 前端客户端 ================
    st.components.v1.html(
        f"""
        <script>
        const ws = new WebSocket("ws://localhost:12393/client-ws");

        ws.onopen = function() {{
            console.log("WS 已连接");
            ws.send(JSON.stringify({{
                type: "bind-user",
                user_name: "{st.session_state.selected_user or ''}"
            }}));
        }};

        ws.onmessage = function(event) {{
            const data = JSON.parse(event.data);
            console.log("收到消息:", data);

            if (data.type === "reminder") {{
                // 把提醒写到 localStorage，让 Streamlit 能取到
                localStorage.setItem("latest_reminder", data.content);
            }}
        }};
        </script>
        """,
        height=0,
    )
# ================ WebSocket 客户端结束 ================



    #print("📂 前端读取提醒缓冲文件路径：", BUFFER_FILE)

    # def fetch_new_reminders(current_user_id):
    #     """从缓冲区取出新提醒，塞到 st.session_state.messages"""
    #     if not os.path.exists(BUFFER_FILE):
    #         print("⚠️ 没有找到缓冲文件")
    #         return
    #     try:
    #         with open(BUFFER_FILE, "r", encoding="utf-8") as f:
    #             buffer = json.load(f)
    #         #print("读取到缓冲区：", buffer)
    #     except Exception as e:
    #         print("读取缓冲文件失败：", e)
    #         buffer = []

    #     if not buffer:
    #         return

    #     for rem in buffer:
    #         if rem["user_id"] == current_user_id:
    #             if "messages" not in st.session_state:
    #                 st.session_state.messages = []
    #             exists = any(m.get("content") == rem["content"] for m in st.session_state.messages)
    #             if not exists:
    #                 st.session_state.messages.append({
    #                     "role": "assistant",
    #                     "content": rem["content"],
    #                     "timestamp": rem["timestamp"]
    #                 })
    #                 print("🎯 已追加提醒到消息列表：", rem["content"])
    #                 st.rerun()   # 🔥 立刻刷新界面

    # 每次渲染前先拉取提醒
    if st.session_state.selected_user:
        user_id = get_user_id_by_name(st.session_state.selected_user)
        latest_reminder = st.query_params.get("latest_reminder")
#latest_reminder = st.experimental_get_query_params().get("latest_reminder")

        # 每次刷新从 localStorage 读取提醒
        st.components.v1.html(
            """
            <script>
                if (localStorage.getItem("latest_reminder")) {
                    const reminder = localStorage.getItem("latest_reminder");
                    const url = new URL(window.location.href);
                    url.searchParams.set("latest_reminder", reminder);
                    window.location.href = url.toString();
                    localStorage.removeItem("latest_reminder");
                }
            </script>
            """,
            height=0
        )

        if latest_reminder:
            reminder_text = latest_reminder[0]
            st.session_state.messages.append({
                "role": "assistant",
                "content": "⏰ 提醒：" + reminder_text
            })

        #fetch_new_reminders(user_id)

    # 自动刷新（JS，每 5 秒刷新一次）
    st.components.v1.html(
        """
        <script>
        const REFRESH_INTERVAL = 5000; // 5秒
        if (!window.__streamlit_auto_refresh_set) {
            window.__streamlit_auto_refresh_set = true;
            setInterval(() => {
                if (document.visibilityState === "visible") {
                    window.location.reload();
                }
            }, REFRESH_INTERVAL);
        }
        </script>
        """,
        height=0,
    )

    # 侧边栏配置
    with st.sidebar:
        st.markdown("<h3>💬 对话设置</h3>", unsafe_allow_html=True)
        st.markdown("---")

        # ---------- RAG 设置 ----------
        st.subheader("RAG 设置")
        rag_enabled = st.checkbox("启用 RAG 检索", value=st.session_state.rag_enabled)

        if rag_enabled != st.session_state.rag_enabled:
            st.session_state.rag_enabled = rag_enabled
            st.rerun()

        if st.session_state.rag_enabled:
            rag_threshold = st.slider(
                "相似度阈值",
                0.1, 0.9,
                st.session_state.rag_threshold,
                0.05
            )
            if rag_threshold != st.session_state.rag_threshold:
                st.session_state.rag_threshold = rag_threshold
                st.rerun()

        st.markdown("---")

        # ---------- 新建对话 ----------
        st.subheader("对话管理")
        if st.button("🆕 新建对话", use_container_width=True):
            create_new_conversation()   # 已有的自动保存逻辑

        st.markdown("---")

        # ---------- 新增：退出按钮 ----------
        if st.button("🚪 退出对话", use_container_width=True):
            # 自动保存对话（调用你已有的保存函数）
            create_new_conversation()

            # 返回dashboard页
            st.session_state.page = "dashboard"
            st.rerun()
   
    # 主界面
    if st.session_state.selected_conversation is not None:
        selected_conv = st.session_state.conversation_history[st.session_state.selected_conversation]
        st.subheader(f"📜 对话历史: {selected_conv['start_time']} 至 {selected_conv['end_time']}")
        st.markdown("---")
        for message in selected_conv["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"], unsafe_allow_html=True)
        if st.button("🔙 返回当前对话", use_container_width=True):
            st.session_state.selected_conversation = None
            st.rerun()
    else:
        st.subheader(f"💬 正在与 {st.session_state.selected_user or '未知用户'} 对话")
        st.markdown("---")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if "rag_source" in message and message["rag_source"]:
                    st.markdown(f'<div class="rag-answer">📚 {message["content"]}</div>', unsafe_allow_html=True)
                # elif message["content"].startswith("⏰ 提醒："):
                #     st.markdown(
                #         f'<div style="background-color:#fff3cd; padding:10px; border-radius:4px;">{message["content"]}</div>',
                #         unsafe_allow_html=True
                #     )
                else:
                    st.markdown(message["content"], unsafe_allow_html=True)

        if prompt := st.chat_input("请输入您的问题...", key="user_input"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt, unsafe_allow_html=True)
            # 首次发消息时，记录对话开始时间
            if st.session_state.conversation_start_time is None:
                st.session_state.conversation_start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            selected_user = st.session_state.selected_user
            if selected_user:
                user_id = get_user_id_by_name(selected_user)
                if user_id is not None:
                    # 1. 提取情感需求（列表）
                    emotional_needs = emotion_extractor.extract_needs(prompt)
                    # 2. 直接传递列表给更新函数（而非拼接后的字符串）
                    update_result = update_user_emotional_needs(user_id, emotional_needs)

                    # 同时写入日志表（用对话结束时间，此处先标记，新建对话时统一处理）
                    if st.session_state.conversation_history:
                        # 暂存逻辑，实际新建对话时用结束时间正式记录，也可直接用开始时间，根据需求调整
                        pass  
                    else:
                        pass    #⚠️ 对话历史为空，无法获取时间戳print("")
                    st.session_state.emotional_needs_history.append(emotional_needs)
                    if not update_result:
                        print("⚠️ 情感需求更新失败，请检查数据库操作")
                    else:
                        emotional_needs_str = ", ".join(emotional_needs)
                        st.session_state.emotional_needs_history.append(emotional_needs)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                thinking_html = """
                <div class="markdown-content">
                    <div style="display: flex; align-items: center;">
                        <div class="streamlit-spinner" style="margin-right: 10px;"></div>
                        <span>思考中...</span>
                    </div>
                </div>
                """
                message_placeholder.markdown(thinking_html, unsafe_allow_html=True)

            # 首先尝试使用 RAG 回答（如果启用且包含健康关键字）
            rag_answer = None
            if st.session_state.rag_enabled and is_health_related(prompt):  # 核心修改：增加关键字检测
                try:
                    print(f"检测到健康相关关键字，触发RAG检索...")
                    rag_answer = get_rag_answer_or_fallback(prompt, st.session_state.rag_threshold)
                    if rag_answer and not rag_answer.startswith("❌"):
                        # 成功获取 RAG 回答
                        message_placeholder.markdown(f'<div class="rag-answer">📚 {rag_answer}</div>', unsafe_allow_html=True)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": rag_answer,
                            "rag_source": True  # 标记为 RAG 回答
                        })
                        st.session_state.last_response_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        st.rerun()
                    elif rag_answer and rag_answer.startswith("❌"):
                        st.warning(f"RAG 检索失败: {rag_answer}")
                except Exception as e:
                    st.warning(f"RAG 处理异常: {str(e)}")

            # 如果 RAG 未找到答案或禁用，使用大模型回答
            user_profile = get_user_profile_by_name(st.session_state.selected_user)
            # 动态调整健康信息（仅健康相关问题显示）
            health_info = ""
            if is_health_related(prompt) and user_profile and user_profile.get("dynamic_health"):
                health_info = f"该用户曾经患有以下疾病：{user_profile['dynamic_health']}。请在合适的时机关心用户的健康情况。"

            # 新增：将情感需求加入系统提示词
            profile_str = ""
            if user_profile:
                profile_items = [f"{key}：{value}" for key, value in user_profile.items()]
                profile_str = "以下是该用户的基本资料：\n" + "\n".join(profile_items)

            # 增强提示词：包含情感需求
            emotional_needs_prompt = ""
            if st.session_state.emotional_needs_history:
                latest_needs = st.session_state.emotional_needs_history[-1]
                emotional_needs_prompt = f"用户当前情感需求：{', '.join(latest_needs)}。请根据需求提供相应支持。\n"

            # 如果 RAG 找到了相关信息但生成失败，可以将其加入提示词
            rag_context = ""
            if rag_answer and rag_answer.startswith("❌") and "匹配到最相关段落" in rag_answer:
                # 提取相关信息加入上下文
                rag_context = "\n注意：系统检索到相关健康信息但生成失败，请参考相关知识进行回答。"
####################################################################################################################
            # 新增：提取历史健康问题
            from rag_answer import extract_recent_health_issues  # 导入新函数
            # 从会话状态中获取历史对话
            history_health_issues = extract_recent_health_issues(st.session_state.messages[:-1])  # 排除当前提问

            # 构建历史提醒文本
            history_reminder = ""
            if history_health_issues:
                history_reminder = "\n\n历史健康信息提醒：\n"
                for issue in history_health_issues:
                    history_reminder += f"- 用户之前提到过{issue}，请在回复中适当询问恢复情况\n"

            # 修改 messages 生成部分，加入历史提醒
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT + "\n" +
                profile_str + "\n" + health_info + "\n" + emotional_needs_prompt + 
                rag_context + history_reminder},  # 关键：加入历史提醒
                *[{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages]
            ]
            # 在 messages 生成后添加
            #print("最终系统提示词：", messages[0]["content"])
######################################################################################################################
            #messages = [
             #   {"role": "system", "content": SYSTEM_PROMPT + "\n" +
              #   profile_str + "\n" + health_info + "\n" + emotional_needs_prompt + rag_context},
               # *[{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages]
            #]

            # ✅ 修改：调用本地 Ollama API
            try:
                start_time = time.time()
    
                # 学校服务器的 API 格式（与 OpenAI 兼容）
                school_server_payload = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": st.session_state.temperature,
                    "top_p": st.session_state.top_p,
                    "max_tokens": st.session_state.max_tokens,
                    "stream": False
                }
                
                # 添加认证头 ← 就是这里！
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                response = requests.post(f"{base_url}/chat/completions", 
                                    headers=headers,  # ← 在这里使用 headers
                                    json=school_server_payload, 
                                    timeout=60)
                response.raise_for_status()
                result = response.json()
                full_response = result["choices"][0]["message"]["content"].strip()
                
               
                
                end_time = time.time()
                st.session_state.last_response_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                
                message_placeholder.markdown(full_response, unsafe_allow_html=True)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "rag_source": False  # 标记为非 RAG 回答
                })

                response_time = end_time - start_time
                st.markdown(f"<div style='text-align: right;'>响应时间: {response_time:.2f} 秒</div>", unsafe_allow_html=True)

            except Exception as e:
                message_placeholder.error(f"发生错误: {str(e)}")
                st.error(f"Ollama API 请求失败: {str(e)}")