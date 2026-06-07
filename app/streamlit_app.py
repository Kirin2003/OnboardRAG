"""
OnboardRAG — Streamlit 前端页面（阶段二 MVP）。

启动方式:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.query_rewriter import QueryRewriter
from src.retriever import Retriever
from src.reranker import Reranker
from src.generator import Generator
from src.config import RETRIEVAL_TOP_K


st.set_page_config(
    page_title="OnboardRAG - 新员工入职助手",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 OnboardRAG — 新员工入职知识助手")
st.caption("基于企业文档的智能问答，回答均附带来源引用")

# ── 侧边栏：参数配置 ────────────────────────────
with st.sidebar:
    st.header("⚙️ 检索参数")
    top_k = st.slider("返回 chunk 数", 3, 20, RETRIEVAL_TOP_K)

    st.header("💡 示例问题")
    examples = [
        "入职第一天需要做什么？",
        "试用期怎么转正？",
        "OA 系统怎么登录？",
        "VPN 连上了但是打不开内网怎么办？",
        "怎么重置密码？",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.question = ex

# ── 主区域 ────────────────────────────────────
question = st.text_input(
    "👇 请输入你的问题",
    value=st.session_state.get("question", ""),
    placeholder="例如：试用期怎么转正？",
    key="question_input",
)

# 回车或按钮触发
col1, col2 = st.columns([1, 5])
with col1:
    ask_clicked = st.button("🔍 查询", type="primary")
with col2:
    show_chunks = st.checkbox("显示检索到的原始 chunks", value=False)

# 合并触发条件：按钮点击 或 选择示例问题后自动查询
# 示例问题点击后会设置 session_state.question，需要检测变化后自动查询
auto_question = st.session_state.get("question", "")
if auto_question and auto_question != st.session_state.get("_last_question", ""):
    st.session_state["_last_question"] = auto_question
    ask_clicked = True
    question = auto_question

if ask_clicked and question.strip():
    query = question.strip()

    with st.spinner("🔍 正在检索相关文档..."):
        try:
            # Step 1: Query Rewrite
            rewriter = QueryRewriter()
            rewritten = rewriter.rewrite(query)

            # Step 2: 混合检索
            retriever = Retriever()
            chunks = retriever.retrieve(rewritten, top_k=top_k)

            if not chunks:
                st.warning("未找到相关文档，请尝试换个问法。")
                st.stop()

            # Step 3: Reranker
            reranker = Reranker()
            chunks = reranker.rerank(rewritten, chunks, top_k=top_k)

            # Step 4: LLM 生成
            generator = Generator()
            answer, sources = generator.generate(rewritten, chunks[:top_k])

        except Exception as e:
            st.error(f"查询出错: {e}")
            raise

    # ── 显示答案 ──
    st.markdown("---")
    st.markdown("### 🤖 答案")
    st.markdown(answer)

    # ── 显示参考来源 ──
    if sources:
        st.markdown("---")
        st.markdown("### 📚 参考来源")
        for i, src in enumerate(sources, 1):
            page_start = src.get("page_start", 0)
            page_end = src.get("page_end", 0)
            page_info = (f"第{page_start}页" if page_start == page_end
                         else f"第{page_start}-{page_end}页")
            with st.expander(
                f"[{i}] {src['doc_title']} — {src['source_file']} ({page_info})"
            ):
                st.caption(f"chunk_id: {src['chunk_id']}")
                st.text(src.get("text", "")[:500])

    # ── 显示原始 chunks（可选） ──
    if show_chunks and chunks:
        st.markdown("---")
        st.markdown("### 🔍 检索到的原始 Chunks")
        for i, c in enumerate(chunks[:top_k], 1):
            text_preview = c.get("body_text", c.get("text", ""))[:300]
            score = c.get("score", 0)
            with st.expander(
                f"[{i}] {c['doc_title']} "
                f"第{c.get('page_start',0)}-{c.get('page_end',0)}页 "
                f"(score: {score:.4f})"
            ):
                st.text(text_preview)
