# OnboardRAG：企业新员工入职知识助手

## 一、业务背景

我要实现一个小而垂直的 RAG 项目，项目名称为 **OnboardRAG：企业新员工入职知识助手**。

业务场景不是泛企业知识库，而是专门面向新员工入职早期的高频问题。新员工刚入职时，经常会反复询问 HR、IT、行政人员或 mentor，例如：

* 入职流程是什么？
* 试用期怎么转正？
* 请假、考勤、工资条、社保公积金规则是什么？
* OA 系统怎么登录？
* 移动办公 App 怎么使用？
* VPN 怎么连接？
* 公司内网怎么访问？
* 忘记密码、系统进不去、VPN 连不上怎么办？

这些问题通常分散在员工手册、OA 系统使用手册、企业内部移动办公手册、VPN 使用手册等 PDF 文档中。传统方式依赖人工答疑，重复性强，浪费 HR、IT 和 mentor 的时间。

本项目目标是构建一个 **面向新员工 onboarding 场景的垂直 RAG 问答助手**，从少量 PDF 文档中检索相关依据，并生成带来源引用的回答。

本项目不做全公司知识库，不涉及权限隔离，不涉及真实上线流量，只做一个试点原型，并通过离线测试集评估效果。

---

## 二、项目数据范围

当前只处理以下几类 PDF 文档：

1. 员工手册

   * 覆盖入职、转正、考勤、休假、薪酬福利、保密制度等。

2. OA 系统使用手册

   * 覆盖 OA 登录、审批中心、请假、报销、密码重置、流程查询等。

3. 企业内部移动办公使用手册

   * 覆盖移动端工作台、审批、通知、会议室、消息、移动端流程查看等。

4. VPN 使用手册

   * 覆盖 VPN 客户端安装、服务器配置、登录认证、访问内网资源、连接失败排查等。

这些文档默认都是所有员工可见，不需要实现用户权限控制。

---

## 三、项目分阶段实施

本项目分两个阶段实施，先跑通数据入库，再做检索生成。

### 阶段一（✅ 已完成）：离线数据入库

将 PDF 文档解析、清洗、切分、向量化后写入向量数据库。不涉及查询、检索或答案生成。

**已完成内容：**

* `src/pdf_loader.py` — PyMuPDF 提取 PDF 文本
* `src/cleaner.py` — 文本清洗
* `src/chunker.py` — 语义 chunk 切分（300–800 字，50–100 字 overlap）
* `src/embedder.py` — BAAI/bge-large-zh-v1.5 向量化
* `src/milvus_store.py` — Milvus Lite 入库
* `scripts/build_index.py` — 一键入库脚本

### 阶段二（当前 🔥）：在线检索与生成

在阶段一基础上实现最小可运行 RAG 问答链路：

1. **query rewrite（查询改写）** — 先做最小版本（直接返回原 query），保留接口后续接 LLM
2. **混合检索** — dense 向量检索（Milvus）+ BM25 关键词检索（rank_bm25）→ RRF 合并
3. **reranker 重排序** — 可插拔接口，默认关闭，不影响 MVP 跑通
4. **LLM 生成带来源引用的回答** — OpenAI-compatible API，要求回答中引用来源 [1]、[2]
5. **CLI 问答脚本** — `scripts/ask.py`，支持命令行单轮问答
6. **Streamlit 页面（可选）** — 在 CLI 跑通后做最小 UI

阶段二先做 MVP，**不实现**以下内容：
* 复杂 agent / 工具调用
* 多轮对话 / 对话历史
* 用户权限控制
* 复杂 onboarding checklist 生成

---

## 四、最小可运行 RAG 链路

### 阶段一（✅）：离线入库

```text
PDF 文档
  ↓
文本提取
  ↓
文本清洗
  ↓
chunk 切分
  ↓
metadata 标注
  ↓
embedding 向量化
  ↓
写入 Milvus
```

### 阶段二（🔥 当前）：在线检索与生成

```text
用户提问
  ↓
查询改写（MVP：直接返回原 query）
  ↓
┌──────────────────────────────────────────┐
│  dense 向量检索（Milvus）                 │
│  BM25 关键词检索（rank_bm25 + jieba）     │
│  → RRF 合并                              │
└──────────────────────────────────────────┘
  ↓
重排序（MVP：默认关闭，可插拔 BGE-reranker）
  ↓
LLM 生成（OpenAI-compatible API）
  ↓
返回：答案 + 来源引用 [1][2]...
```

---

## 五、核心功能需求

### 1. PDF 文本提取

使用 `PyMuPDF` 读取 PDF。

每页提取：

* page number
* text
* source file name
* doc title
* document category

文档 category 可以根据文件名判断：

* employee_handbook
* oa_manual
* mobile_office_manual
* vpn_manual

如果 PDF 有目录或章节标题，可以尽量保留；如果没有，至少保留页码。

---

### 2. 文本清洗

实现简单但实用的清洗逻辑：

* 去除多余空行
* 去除重复空格
* 去除页眉页脚中的无用内容
* 去除明显的页码噪声
* 合并异常断行
* 保留标题、编号、步骤、FAQ

不需要做复杂 OCR。假设 PDF 是可复制文本 PDF。如果遇到扫描版 PDF，可以先跳过。

---

### 3. Chunk 切分

不要把整本 PDF 直接入库。需要切成 chunk。

最小实现：

* 按页读取文本；
* 再按段落或固定长度切分；
* 每个 chunk 控制在中文 300–800 字左右；
* overlap 设为 50–100 字；
* 尽量不要切断完整的操作步骤或 FAQ。

每个 chunk 至少包含以下字段：

```json
{
  "chunk_id": "vpn_manual_p3_c1",
  "text": "chunk text",
  "doc_title": "VPN 使用手册",
  "source_file": "vpn_manual.pdf",
  "category": "vpn_manual",
  "page_start": 3,
  "page_end": 3
}
```

---

### 4. 向量化

使用中文友好的 embedding 模型。

* `BAAI/bge-large-zh-v1.5`（推荐，效果比 small 好，本地 GPU 轻松运行）

用 `sentence-transformers` 加载模型并生成 dense embedding。

> 为什么选 large 而不是 small：当前开发机有 NVIDIA RTX 2000 Ada (8GB VRAM)，large 模型推理仅需 ~2GB 显存，性价比最高。如果开发机资源紧张，可降级为 `bge-small-zh-v1.5`。

---

### 5. 向量数据库

使用 **Milvus Lite** 作为向量数据库（Milvus 的嵌入式版本）。

选择 Milvus Lite 的原因：

* 零依赖：`pip install pymilvus` 即用，不需要 Docker
* 本地文件存储：类似 SQLite，数据存为一个本地数据库文件
* API 兼容：与 Milvus Standalone / Milvus Cloud 完全一致，未来可无缝升级
* 适合原型：135 页 PDF、几千条 chunk 完全够用

#### 混合检索设计（阶段二实现）

本项目的混合检索方案不依赖特定 embedding 模型：

```
方案（bge-large + rank_bm25）：
  bge-large-zh-v1.5 → dense embedding → Milvus 向量检索 ──┐
                                                            ├── RRF 合并排序 → 最终结果
  原始文本 → rank_bm25 本地计算 → BM25 关键词检索 ────────┘
```

> 为什么不用 bge-m3？bge-m3 的 sparse vector 确实可以和 dense vector 做更好的语义对齐，但对本项目来说 bge-large + rank_bm25 足够，且 bge-large 模型更小（326M vs 568M），显存友好。


### 6. 混合检索（`src/retriever.py`）

阶段二新增，实现 dense + BM25 混合检索。

**Dense 检索：**
- 使用阶段一相同的 `BAAI/bge-large-zh-v1.5` 模型对用户 query 编码
- 从 Milvus Lite 执行向量相似度搜索（`search()`）
- 返回 top_k 候选 chunk

**BM25 关键词检索：**
- 从 `data/processed/chunks.jsonl` 加载所有 chunk 文本
- 使用 `jieba` 分词 + `rank_bm25` 建立 BM25 索引
- 对用户 query 进行关键词检索

**RRF 合并：**
- 将 dense 结果和 BM25 结果用 Reciprocal Rank Fusion 合并排序
- 返回最终 top_k chunk

**返回字段：**
```json
{
  "chunk_id": "vpn_manual_p3_c1",
  "text": "chunk text...",
  "doc_title": "VPN 使用手册",
  "source_file": "vpn_manual.pdf",
  "category": "vpn_manual",
  "page_start": 3,
  "page_end": 3,
  "score": 0.85
}
```

### 7. 查询改写（`src/query_rewriter.py`）

MVP 版本：默认直接返回原 query，不做改写。

保留 `rewrite(query: str) -> str` 接口，后续可接入 LLM 做查询扩展、同义词替换、多轮对话上下文改写等。

### 8. 重排序（`src/reranker.py`）

可插拔接口设计：

- 环境变量 `ENABLE_RERANKER` 控制是否启用（默认 `false`）
- 当 `ENABLE_RERANKER=false` 时，直接返回 RRF 排序结果
- 当 `ENABLE_RERANKER=true` 时，加载 `BAAI/bge-reranker-base` 对候选 chunks 做重排序
- MVP 阶段保持关闭，不影响主链路

接口：
```python
def rerank(query: str, chunks: list[dict], top_k: int = 10) -> list[dict]:
    ...
```

### 9. 答案生成（`src/generator.py`）

使用 OpenAI-compatible API 调用 LLM 生成答案。

**配置（从环境变量读取）：**
- `LLM_BASE_URL` — API 地址
- `LLM_API_KEY` — API 密钥
- `LLM_MODEL` — 模型名称

**Prompt 要求：**
- 只能基于给定的上下文 chunks 回答问题
- 如果上下文中信息不足，回答"根据当前文档未找到明确依据"
- 回答中必须使用 `[1]`、`[2]` 等形式引用来源 chunk 编号
- 回答末尾附参考来源列表，包含 doc_title、source_file、page_start/page_end、chunk_id

### 10. CLI 问答脚本（`scripts/ask.py`）

支持命令行运行：

```bash
python scripts/ask.py "VPN 连上了但是打不开内网怎么办？"
```

输出：
- 用户问题
- LLM 生成的答案（含来源引用编号）
- 参考来源列表（doc_title、source_file、page_start/page_end、chunk_id）

### 11. Streamlit 页面（`app/streamlit_app.py`，可选）

在 CLI 跑通后实现最小 UI：
- 问题输入框
- 答案展示区（渲染引用标记）
- 可折叠的引用来源列表
- 可折叠的原始检索 chunks 展示

---

## 六、技术栈

语言：Python

推荐技术栈：

```text
PDF 解析：
- PyMuPDF

文本处理：
- re
- pathlib
- json
- pandas

Embedding：
- BAAI/bge-large-zh-v1.5（推荐）
- BAAI/bge-small-zh-v1.5（备选）

向量数据库：
- Milvus Lite（pymilvus 内嵌，无需 Docker）

混合检索（阶段二）：
- rank_bm25

后端（阶段二）：
- FastAPI，可选

前端（阶段二）：
- streamlit

配置：
- python-dotenv
- pydantic
```

---

## 七、建议项目目录结构

```text
onboard-rag/
  data/
    raw_pdfs/              # 原始 PDF 文档
    processed/             # 处理后的中间文件
      chunks.jsonl         # 切分后的 chunk 文本（BM25 检索用）
    onboard_rag.db/        # Milvus Lite 数据库文件（自动创建）
    eval/                  # 离线评估
      eval_queries.csv

  src/
    config.py              # 全局配置
    pdf_loader.py          # PDF 文本提取（阶段一）
    cleaner.py             # 文本清洗（阶段一）
    chunker.py             # chunk 切分（阶段一）
    embedder.py            # embedding 向量化（阶段一）
    milvus_store.py        # Milvus Lite 入库（阶段一）
    retriever.py           # 混合检索：dense + BM25 + RRF（阶段二）
    query_rewriter.py      # 查询改写（阶段二，MVP：直接返回原 query）
    reranker.py            # 重排序（阶段二，MVP：默认关闭）
    generator.py           # LLM 答案生成（阶段二）

  app/
    streamlit_app.py       # Streamlit 前端页面（阶段二，可选）

  scripts/
    build_index.py         # 一键入库（阶段一）
    build_chunks.py        # 单独生成 chunks.jsonl
    ask.py                 # CLI 问答脚本（阶段二）
    run_eval.py            # 离线评估

  requirements.txt
  README.md
  .env.example
```

---

## 八、最小运行命令

### 安装依赖

```bash
pip install -r requirements.txt
```

### 阶段一：离线入库

```bash
# 解析 PDF、切 chunk、向量化、入库
python scripts/build_index.py
```

### 阶段二：问答

```bash
# 1. 配置 LLM 环境变量（编辑 .env，参考 .env.example）
cp .env.example .env

# 2. 确保索引已建立（阶段一已完成）
# python scripts/build_index.py

# 3. CLI 单轮问答
python scripts/ask.py "入职第一天需要做什么？"
python scripts/ask.py "VPN 连上了但是打不开内网怎么办？"

# 4. 启动 Streamlit 页面（可选）
streamlit run app/streamlit_app.py
```

---

## 九、项目边界说明

本项目是试点原型，不做生产级复杂能力：

* 不接入真实企业 SSO；
* 不做权限隔离；
* 不做真实上线用户统计；
* 不做文档实时增量更新；
* 不做 OCR；
* 不做多模态截图理解。

### 阶段一（✅ 已完成）

实现离线数据入库链路（PDF → 文本 → chunk → embedding → Milvus Lite），亮点：

1. 小而垂直的 onboarding 场景；
2. PDF 文档解析与语义 chunk 切分；
3. Milvus Lite 零依赖向量数据库。

### 阶段二（🔥 当前）

在阶段一基础上增加检索与生成能力，MVP 范围：

1. query rewrite（最小实现，保留接口）；
2. 混合检索：dense（Milvus）+ BM25（rank_bm25 + jieba）→ RRF 合并；
3. reranker 可插拔接口（默认关闭）；
4. LLM 生成带 `[1]` `[2]` 来源引用的回答；
5. CLI 问答脚本 + 可选 Streamlit 页面。

**MVP 不做：**
- 复杂 agent / 工具调用
- 多轮对话 / 对话历史
- 用户权限控制
- 复杂 onboarding checklist 自动生成
