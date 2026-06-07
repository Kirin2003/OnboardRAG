"""
答案生成模块：使用 OpenAI-compatible API 调用 LLM 生成带来源引用的回答。

Usage:
    from src.generator import Generator
    gen = Generator()
    answer, sources = gen.generate(query, chunks)
"""

from openai import OpenAI

from src.config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL


_SYSTEM_PROMPT = """你是一个企业新员工入职知识助手。你的任务是根据给定的文档片段回答用户的问题。

请严格遵守以下规则：
1. 只能基于下面提供的【参考文档】来回答问题，不要使用你自己的知识。
2. 如果【参考文档】中没有足够的信息回答用户的问题，请直接说"根据当前文档未找到明确依据"。
3. 回答时，必须使用 [1]、[2] 等编号引用具体的参考文档片段。
4. 保持回答简洁、清晰，使用中文。
5. 如果问题涉及操作步骤，请按顺序列出来。
6. 最后用"参考来源："列出引用的文档，格式为：文档名称 第X页"""


def _build_context(chunks: list[dict]) -> str:
    """将 chunk 列表构建为 LLM 上下文文本。

    每个 chunk 格式：
    [1] 文档：{doc_title}
       正文：{text}
       （来源：{source_file}, 第{page_start}-{page_end}页）
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        # 优先用 body_text，更干净
        text = chunk.get("body_text", chunk.get("text", ""))
        doc_title = chunk.get("doc_title", "未知文档")
        source_file = chunk.get("source_file", "")
        page_start = chunk.get("page_start", 0)
        page_end = chunk.get("page_end", 0)

        if page_start == page_end:
            page_info = f"第{page_start}页"
        else:
            page_info = f"第{page_start}-{page_end}页"

        parts.append(
            f"[{i}] 文档：{doc_title}\n"
            f"    正文：{text}\n"
            f"    （来源：{source_file}, {page_info}）"
        )

    return "\n\n".join(parts)


class Generator:
    """LLM 答案生成器。

    使用 OpenAI-compatible API 调用 LLM，
    基于检索到的 chunks 生成带来源引用的回答。
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        base_url = base_url or LLM_BASE_URL
        api_key = api_key or LLM_API_KEY
        self.model = model or LLM_MODEL

        if not base_url:
            raise ValueError("LLM_BASE_URL 未设置，请在 .env 中配置")
        if not api_key:
            raise ValueError("LLM_API_KEY 未设置，请在 .env 中配置")

        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def generate(
        self,
        query: str,
        chunks: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> tuple[str, list[dict]]:
        """根据问题和检索到的 chunks 生成答案。

        Args:
            query: 用户问题
            chunks: 检索到的 chunk 列表（已排序）
            temperature: LLM 温度
            max_tokens: 最大输出 token 数

        Returns:
            (answer, sources) 元组
            - answer: LLM 生成的答案文本（含引用标记 [1][2]）
            - sources: 被引用的 chunk 列表（与编号对应，即入参 chunks）
        """
        if not chunks:
            return "根据当前文档未找到明确依据", []

        context = _build_context(chunks)

        user_message = (
            f"【参考文档】\n{context}\n\n"
            f"【用户问题】\n{query}\n\n"
            f"请根据以上参考文档回答用户问题。"
        )

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        answer = response.choices[0].message.content.strip()

        # sources 与 chunks 一一对应（编号 [1] [2] 即 chunks 的索引+1）
        sources = [
            {
                "chunk_id": c.get("chunk_id", ""),
                "doc_title": c.get("doc_title", "未知文档"),
                "source_file": c.get("source_file", ""),
                "page_start": c.get("page_start", 0),
                "page_end": c.get("page_end", 0),
                "text": c.get("body_text", c.get("text", ""))[:200],
            }
            for c in chunks
        ]

        return answer, sources
