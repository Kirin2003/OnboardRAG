"""
查询改写模块（MVP：最小实现，直接返回原 query）。

后续可接入 LLM 做查询扩展、同义词替换、多轮对话上下文改写等。

Usage:
    from src.query_rewriter import QueryRewriter
    rewriter = QueryRewriter()
    rewritten = rewriter.rewrite("VPN 连不上怎么办？")
"""


class QueryRewriter:
    """查询改写器。

    MVP 版本不做任何改写，直接返回原 query。
    保留接口供后续扩展（LLM 改写、同义词扩展、上下文注入等）。
    """

    def rewrite(self, query: str) -> str:
        """对用户查询进行改写（MVP：直接返回原 query）。

        Args:
            query: 用户原始查询文本

        Returns:
            改写后的查询文本
        """
        return query.strip()
