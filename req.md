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

### 阶段一（当前）：离线数据入库

将 PDF 文档解析、清洗、切分、向量化后写入向量数据库。不涉及查询、检索或答案生成。

### 阶段二（后续）：在线检索与生成

在阶段一基础上实现：

* query rewrite（查询改写）
* dense 向量检索（Milvus）+ BM25 关键词检索（rank_bm25）→ RRF 合并
* reranker 重排序
* LLM 生成带来源引用的回答
* 复杂问题生成 onboarding checklist

---

## 四、最小可运行 RAG 链路（阶段一）

阶段一只实现离线数据入库 pipeline：

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
    raw_pdfs/
      

    processed/
      chunks.jsonl

    eval/
      eval_queries.csv

  src/
    config.py
    pdf_loader.py
    cleaner.py
    chunker.py
    embedder.py
    milvus_store.py
    

  app/
    streamlit_app.py

  scripts/
    build_index.py
    run_eval.py

  requirements.txt
  README.md
  .env.example
```

---

## 八、最小运行命令

阶段一只需两行命令：

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 解析 PDF、切 chunk、向量化、入库
python scripts/build_index.py
```

> 使用 Milvus Lite 后不需要 Docker。`pymilvus` 自带嵌入式数据库，`build_index.py` 第一次运行时会自动创建本地数据库文件。

---

## 九、项目边界说明

本项目是试点原型，不做生产级复杂能力：

* 不接入真实企业 SSO；
* 不做权限隔离；
* 不做真实上线用户统计；
* 不做文档实时增量更新；
* 不做 OCR；
* 不做多模态截图理解。

### 阶段一范围（当前）

只实现离线数据入库链路（PDF → 文本 → chunk → embedding → Milvus Lite），亮点：

1. 小而垂直的 onboarding 场景；
2. PDF 文档解析与语义 chunk 切分；
3. Milvus Lite 零依赖向量数据库。

### 阶段二范围（后续）

在阶段一基础上增加检索与生成能力：

1. query rewrite；
2. 关键词检索 + 向量检索 hybrid search（bge-large + rank_bm25）；
3. reranker 重排序；
4. 来源引用；
5. 对复杂问题生成 onboarding checklist。
