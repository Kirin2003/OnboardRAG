"""
Unit tests for src/eval_metrics.py — evidence-level retrieval metrics.
"""

import pytest
from src.eval_metrics import (
    normalize_text,
    char_3grams,
    char_3gram_containment,
    rouge_l_recall,
    partial_ratio,
    exact_evidence_match,
    fuzzy_evidence_match,
    evidence_hit_at_k,
    evidence_mrr,
    evidence_recall_at_k,
    get_evidence_units,
    build_retrieved_context,
    match_evidence_unit_in_context,
    compute_evidence_unit_metrics,
    make_exact_matcher,
    make_fuzzy_matcher,
    _lcs_length,
)


# ═══════════════════════════════════════════════════════════════
# normalize_text
# ═══════════════════════════════════════════════════════════════

class TestNormalizeText:
    def test_removes_whitespace(self):
        assert normalize_text("忘 记 打 卡") == "忘记打卡"
        assert normalize_text("VPN  连接\n失败") == "vpn连接失败"

    def test_removes_punctuation(self):
        assert normalize_text("新员工入职需要准备哪些材料？") == "新员工入职需要准备哪些材料"
        assert normalize_text("【右上角头像】—【个人中心】") == "右上角头像个人中心"
        assert normalize_text("请准备：《录用通知书》、身份证原件及复印件。") == "请准备录用通知书身份证原件及复印件"

    def test_lowercase(self):
        assert normalize_text("VPN") == "vpn"
        assert normalize_text("OA系统") == "oa系统"

    def test_nfkc_normalization(self):
        # 全角字母 → 半角
        assert normalize_text("ＡＢＣ") == "abc"
        # 全角数字 → 半角
        assert normalize_text("１２３") == "123"

    def test_keeps_chinese_english_digits(self):
        result = normalize_text("统一账号@2024年.vpn")
        assert result == "统一账号2024年vpn"


# ═══════════════════════════════════════════════════════════════
# char_3grams
# ═══════════════════════════════════════════════════════════════

class TestChar3grams:
    def test_basic(self):
        grams = char_3grams("abcdef")
        assert grams == {"abc", "bcd", "cde", "def"}

    def test_chinese(self):
        grams = char_3grams("怎么激活")
        assert grams == {"怎么激", "么激活"}

    def test_short_text(self):
        assert char_3grams("ab") == set()
        assert char_3grams("") == set()

    def test_chinese_single(self):
        assert char_3grams("打卡") == set()


# ═══════════════════════════════════════════════════════════════
# char_3gram_containment
# ═══════════════════════════════════════════════════════════════

class TestChar3gramContainment:
    def test_full_match(self):
        assert char_3gram_containment("abcdef", "abcdefxyz") == 1.0
        assert char_3gram_containment("怎么激活", "怎么激活账号") == 1.0

    def test_partial_match(self):
        evidence = "abcdef"
        chunk = "abcxyz"
        # evidence 3-grams: abc, bcd, cde, def → 4
        # chunk 3-grams: abc, bcx, cxy, xyz → 4
        # overlap: abc → 1
        assert char_3gram_containment(evidence, chunk) == pytest.approx(1.0 / 4)

    def test_no_match(self):
        assert char_3gram_containment("abc", "xyz") == 0.0

    def test_empty_evidence(self):
        assert char_3gram_containment("", "anything") == 0.0


# ═══════════════════════════════════════════════════════════════
# _lcs_length
# ═══════════════════════════════════════════════════════════════

class TestLCSLength:
    def test_identical(self):
        assert _lcs_length("abc", "abc") == 3

    def test_subsequence(self):
        assert _lcs_length("abc", "axbyc") == 3

    def test_no_common(self):
        assert _lcs_length("abc", "xyz") == 0

    def test_empty(self):
        assert _lcs_length("", "abc") == 0
        assert _lcs_length("abc", "") == 0

    def test_chinese(self):
        assert _lcs_length("怎么激活账号", "激活账号") == 4


# ═══════════════════════════════════════════════════════════════
# rouge_l_recall
# ═══════════════════════════════════════════════════════════════

class TestRougeLRecall:
    def test_full_match(self):
        assert rouge_l_recall("abc", "xyzabc123") == 1.0

    def test_partial(self):
        recall = rouge_l_recall("abcdef", "axbycz")
        # LCS("abcdef", "axbycz") = "abc" = 3
        # recall = 3/6 = 0.5
        assert recall == pytest.approx(0.5)

    def test_no_match(self):
        assert rouge_l_recall("abc", "xyz") == 0.0

    def test_empty_evidence(self):
        assert rouge_l_recall("", "abc") == 0.0


# ═══════════════════════════════════════════════════════════════
# partial_ratio
# ═══════════════════════════════════════════════════════════════

class TestPartialRatio:
    def test_exact_substring(self):
        # "忘记密码" 是较长 chunk 的子串 → 滑动窗口应找到 1.0
        score = partial_ratio("忘记密码", "如何找回忘记密码的步骤")
        assert score == pytest.approx(1.0)

    def test_near_match(self):
        # "忘记密码怎么办" 在 "怎么找回忘记密码" 中滑动 → 应有部分 3-gram 匹配
        score = partial_ratio("忘记密码怎么办", "怎么找回忘记密码的方法")
        assert score > 0.0

    def test_no_match(self):
        score = partial_ratio("abcdefghij", "xyz123")
        assert score == 0.0

    def test_short_text(self):
        # 短于 3 字符，降级为字符级匹配
        score = partial_ratio("ab", "abcde")
        assert score > 0.0


# ═══════════════════════════════════════════════════════════════
# exact_evidence_match
# ═══════════════════════════════════════════════════════════════

class TestExactEvidenceMatch:
    def test_exact_substring_match(self):
        assert exact_evidence_match(
            "忘记打卡怎么办",
            "2.3 未打卡：员工没有在考勤系统亲自打卡。忘记打卡怎么办？员工须在当月提交说明。"
        ) is True

    def test_normalized_match(self):
        # 标点、空白差异应被 normalize 消除
        assert exact_evidence_match(
            "员工须在当月考勤周期内及时提交未打卡说明",
            "员工须在当月考勤周期内及时提交未打卡说明。"
        ) is True

    def test_no_match(self):
        assert exact_evidence_match(
            "今天天气很好",
            "员工须在当月考勤周期内及时提交未打卡说明"
        ) is False


# ═══════════════════════════════════════════════════════════════
# fuzzy_evidence_match
# ═══════════════════════════════════════════════════════════════

class TestFuzzyEvidenceMatch:
    def test_exact_triggers_fuzzy(self):
        is_match, scores = fuzzy_evidence_match(
            "忘记打卡怎么办",
            "未打卡：忘记打卡怎么办？员工须在当月提交说明。"
        )
        assert is_match is True
        assert scores["exact_match"] is True

    def test_high_overlap_matches(self):
        # "试用期内员工辞职应至少提前三日提出书面离职申请"
        # vs chunk containing similar text
        is_match, scores = fuzzy_evidence_match(
            "试用期内员工辞职应至少提前三日提出书面离职申请",
            "试用期内员工辞职应至少提前三日提出书面离职申请，转正后员工应至少提前三十日。"
        )
        assert is_match is True

    def test_no_match_on_unrelated(self):
        is_match, scores = fuzzy_evidence_match(
            "VPN客户端在哪里下载安装",
            "员工须在当月考勤周期内及时提交未打卡说明"
        )
        assert is_match is False

    def test_returns_scores_dict(self):
        _, scores = fuzzy_evidence_match("abc", "xyz")
        assert "containment" in scores
        assert "rouge_l" in scores
        assert "partial_ratio" in scores
        assert "exact_match" in scores


# ═══════════════════════════════════════════════════════════════
# evidence_hit_at_k
# ═══════════════════════════════════════════════════════════════

class TestEvidenceHitAtK:
    def _make_chunks(self, texts):
        return [{"body_text": t, "source_file": "test.pdf"} for t in texts]

    def test_hit_single_evidence(self):
        chunks = self._make_chunks(["无关内容", "VPN可以远程访问内网", "其他"])
        evidence = [{"quote": "VPN可以远程访问内网"}]
        exact_fn = make_exact_matcher()

        hit, best = evidence_hit_at_k(chunks, evidence, 2, exact_fn)
        assert hit is True
        assert best["rank"] == 2

    def test_miss_single_evidence(self):
        chunks = self._make_chunks(["无关", "其他"])
        evidence = [{"quote": "VPN可以远程访问内网"}]
        exact_fn = make_exact_matcher()

        hit, best = evidence_hit_at_k(chunks, evidence, 2, exact_fn)
        assert hit is False

    def test_k_limit(self):
        chunks = self._make_chunks(["A", "B", "目标内容"])
        evidence = [{"quote": "目标内容"}]
        exact_fn = make_exact_matcher()

        hit, _ = evidence_hit_at_k(chunks, evidence, 2, exact_fn)
        assert hit is False  # k=2, 目标在第3个

        hit, _ = evidence_hit_at_k(chunks, evidence, 3, exact_fn)
        assert hit is True

    def test_multi_evidence_any_hit(self):
        chunks = self._make_chunks(["无关", "包含第二个证据的文本"])
        evidence = [
            {"quote": "第一个证据"},
            {"quote": "第二个证据"},
        ]
        exact_fn = make_exact_matcher()

        hit, best = evidence_hit_at_k(chunks, evidence, 2, exact_fn)
        assert hit is True
        assert best["evidence"]["quote"] == "第二个证据"


# ═══════════════════════════════════════════════════════════════
# evidence_mrr
# ═══════════════════════════════════════════════════════════════

class TestEvidenceMRR:
    def _make_chunks(self, texts):
        return [{"body_text": t, "source_file": "test.pdf"} for t in texts]

    def test_first_rank(self):
        chunks = self._make_chunks(["目标内容", "无关"])
        evidence = [{"quote": "目标内容"}]
        exact_fn = make_exact_matcher()

        mrr, best = evidence_mrr(chunks, evidence, exact_fn)
        assert mrr == pytest.approx(1.0)

    def test_second_rank(self):
        chunks = self._make_chunks(["无关", "目标内容"])
        evidence = [{"quote": "目标内容"}]
        exact_fn = make_exact_matcher()

        mrr, best = evidence_mrr(chunks, evidence, exact_fn)
        assert mrr == pytest.approx(0.5)

    def test_no_match(self):
        chunks = self._make_chunks(["A", "B"])
        evidence = [{"quote": "目标内容"}]
        exact_fn = make_exact_matcher()

        mrr, best = evidence_mrr(chunks, evidence, exact_fn)
        assert mrr == 0.0


# ═══════════════════════════════════════════════════════════════
# evidence_recall_at_k
# ═══════════════════════════════════════════════════════════════

class TestEvidenceRecallAtK:
    def _make_chunks(self, texts):
        return [{"body_text": t, "source_file": "test.pdf"} for t in texts]

    def test_all_evidence_found(self):
        chunks = self._make_chunks(["证据A在这里", "证据B在这里"])
        evidence = [{"quote": "证据A"}, {"quote": "证据B"}]
        exact_fn = make_exact_matcher()

        recall = evidence_recall_at_k(chunks, evidence, 2, exact_fn)
        assert recall == pytest.approx(1.0)

    def test_half_evidence_found(self):
        chunks = self._make_chunks(["证据A在这里"])
        evidence = [{"quote": "证据A"}, {"quote": "证据B"}]
        exact_fn = make_exact_matcher()

        recall = evidence_recall_at_k(chunks, evidence, 1, exact_fn)
        assert recall == pytest.approx(0.5)

    def test_no_evidence_found(self):
        chunks = self._make_chunks(["无关"])
        evidence = [{"quote": "证据A"}]
        exact_fn = make_exact_matcher()

        recall = evidence_recall_at_k(chunks, evidence, 1, exact_fn)
        assert recall == 0.0

    def test_empty_evidence_list(self):
        chunks = self._make_chunks(["内容"])
        evidence = []
        exact_fn = make_exact_matcher()

        recall = evidence_recall_at_k(chunks, evidence, 1, exact_fn)
        assert recall == 0.0

    def test_skips_empty_quotes(self):
        chunks = self._make_chunks(["证据内容"])
        evidence = [{"quote": ""}, {"quote": "证据内容"}]
        exact_fn = make_exact_matcher()

        recall = evidence_recall_at_k(chunks, evidence, 1, exact_fn)
        assert recall == pytest.approx(0.5)  # 只有1个有效 evidence，命中


# ═══════════════════════════════════════════════════════════════
# get_evidence_units
# ═══════════════════════════════════════════════════════════════

class TestGetEvidenceUnits:
    def test_from_evidence_units_field(self):
        entry = {
            "evidence_units": ["unit A", "unit B", "unit C"],
            "evidence": [{"quote": "old evidence"}],
        }
        units = get_evidence_units(entry)
        assert units == ["unit A", "unit B", "unit C"]

    def test_fallback_to_evidence(self):
        entry = {
            "evidence": [
                {"doc": "test.pdf", "section": "s1", "quote": "evidence text 1"},
                {"doc": "test.pdf", "section": "s2", "quote": "evidence text 2"},
            ]
        }
        units = get_evidence_units(entry)
        assert units == ["evidence text 1", "evidence text 2"]

    def test_fallback_skips_empty_quotes(self):
        entry = {
            "evidence": [
                {"quote": ""},
                {"quote": "valid evidence"},
            ]
        }
        units = get_evidence_units(entry)
        assert units == ["valid evidence"]

    def test_empty_entry(self):
        assert get_evidence_units({}) == []

    def test_evidence_units_empty_list_uses_fallback(self):
        """空 evidence_units 列表应该 fallback 到 evidence"""
        entry = {
            "evidence_units": [],
            "evidence": [{"quote": "fallback text"}],
        }
        units = get_evidence_units(entry)
        assert units == ["fallback text"]


# ═══════════════════════════════════════════════════════════════
# build_retrieved_context
# ═══════════════════════════════════════════════════════════════

class TestBuildRetrievedContext:
    def test_concatenates_chunks(self):
        chunks = [
            {"body_text": "chunk one text"},
            {"body_text": "chunk two text"},
            {"text": "chunk three fallback"},
        ]
        ctx = build_retrieved_context(chunks, 2)
        assert "chunk one text" in ctx
        assert "chunk two text" in ctx
        assert "chunk three fallback" not in ctx  # k=2

    def test_fallback_to_text_field(self):
        chunks = [{"text": "only text field"}]
        ctx = build_retrieved_context(chunks, 1)
        assert "only text field" in ctx

    def test_empty_chunks(self):
        assert build_retrieved_context([], 5) == ""


# ═══════════════════════════════════════════════════════════════
# match_evidence_unit_in_context
# ═══════════════════════════════════════════════════════════════

class TestMatchEvidenceUnitInContext:
    def test_exact_match_in_context(self):
        exact_fn = make_exact_matcher()
        is_match, scores = match_evidence_unit_in_context(
            "忘记打卡怎么办",
            "chunk A text.\n未打卡：忘记打卡怎么办？员工须提交说明。\nchunk B text.",
            exact_fn,
        )
        assert is_match is True
        assert scores["exact_match"] is True

    def test_unit_spans_two_chunks(self):
        """Evidence unit 跨两个 retrieved chunk 时，拼接后能被命中"""
        exact_fn = make_exact_matcher()
        # Simulating a unit that is split across two chunks
        # (real chunks don't have English filler — just the raw Chinese text)
        is_match, scores = match_evidence_unit_in_context(
            "员工须在当月提交未打卡说明",
            "第二节 考勤制度。未打卡：员工须在当月提交\n未打卡说明，否则视为旷工。员工应遵守考勤制度。",
            exact_fn,
        )
        # After normalization, whitespace is removed, so evidence appears as contiguous substring
        assert is_match is True

    def test_no_match(self):
        exact_fn = make_exact_matcher()
        is_match, _ = match_evidence_unit_in_context(
            "完全不相关的内容",
            "chunk A text.\nchunk B text.",
            exact_fn,
        )
        assert is_match is False

    def test_fuzzy_match_in_context(self):
        fuzzy_fn = make_fuzzy_matcher()
        # "提出" vs "提交" — fuzzy should still match due to high 3-gram overlap
        is_match, scores = match_evidence_unit_in_context(
            "试用期内员工辞职应至少提前三日提出书面离职申请",
            "第五节 离职。试用期内员工辞职应至少提前三日提交书面离职申请，转正后应至少提前三十日。",
            fuzzy_fn,
        )
        assert is_match is True

    def test_empty_inputs(self):
        exact_fn = make_exact_matcher()
        is_match, _ = match_evidence_unit_in_context("", "some context", exact_fn)
        assert is_match is False
        is_match, _ = match_evidence_unit_in_context("some unit", "", exact_fn)
        assert is_match is False


# ═══════════════════════════════════════════════════════════════
# compute_evidence_unit_metrics
# ═══════════════════════════════════════════════════════════════

class TestComputeEvidenceUnitMetrics:
    def _make_chunks(self, texts):
        return [{"body_text": t, "source_file": "test.pdf"} for t in texts]

    def test_all_units_matched(self):
        chunks = self._make_chunks(["证据A和证据B都在这里"])
        units = ["证据A", "证据B"]
        exact_fn = make_exact_matcher()

        result = compute_evidence_unit_metrics(chunks, units, 1, exact_fn)
        assert result["evidence_unit_total"] == 2
        assert result["evidence_unit_hit_count"] == 2
        assert result["evidence_unit_recall"] == 1.0
        assert result["evidence_group_hit"] is True
        assert len(result["matched_units"]) == 2
        assert len(result["missing_units"]) == 0

    def test_partial_units_matched(self):
        chunks = self._make_chunks(["chunk with 证据A only"])
        units = ["证据A", "证据B"]
        exact_fn = make_exact_matcher()

        result = compute_evidence_unit_metrics(chunks, units, 1, exact_fn)
        assert result["evidence_unit_total"] == 2
        assert result["evidence_unit_hit_count"] == 1
        assert result["evidence_unit_recall"] == 0.5
        assert result["evidence_group_hit"] is True
        assert result["matched_units"] == ["证据A"]
        assert result["missing_units"] == ["证据B"]

    def test_no_units_matched(self):
        chunks = self._make_chunks(["无关内容"])
        units = ["证据A"]
        exact_fn = make_exact_matcher()

        result = compute_evidence_unit_metrics(chunks, units, 1, exact_fn)
        assert result["evidence_unit_recall"] == 0.0
        assert result["evidence_group_hit"] is False

    def test_empty_units(self):
        chunks = self._make_chunks(["内容"])
        exact_fn = make_exact_matcher()

        result = compute_evidence_unit_metrics(chunks, [], 1, exact_fn)
        assert result["evidence_unit_total"] == 0
        assert result["evidence_unit_recall"] == 0.0

    def test_unit_across_chunks_with_k(self):
        """Evidence unit 跨两个 chunk 时，k 足够大才能命中"""
        chunks = self._make_chunks([
            "第二节 考勤。未打卡：员工须在当月",
            "提交未打卡说明，否则视为旷工。",
        ])
        units = ["员工须在当月提交未打卡说明"]
        exact_fn = make_exact_matcher()

        # k=1: only first chunk, can't match (missing "提交未打卡说明")
        result_k1 = compute_evidence_unit_metrics(chunks, units, 1, exact_fn)
        assert result_k1["evidence_group_hit"] is False

        # k=2: both chunks concatenated via newline, normalized text matches
        result_k2 = compute_evidence_unit_metrics(chunks, units, 2, exact_fn)
        assert result_k2["evidence_group_hit"] is True
        assert result_k2["evidence_unit_recall"] == 1.0

    def test_old_format_with_evidence_list(self):
        """旧格式只有 evidence（无 evidence_units）的样本仍能评估"""
        chunks = self._make_chunks(["包含证据一的文本", "包含证据二的文本"])
        exact_fn = make_exact_matcher()

        # 模拟旧格式 entry
        evidence_units = get_evidence_units({
            "evidence": [
                {"quote": "证据一"},
                {"quote": "证据二"},
            ]
        })
        assert evidence_units == ["证据一", "证据二"]

        result = compute_evidence_unit_metrics(chunks, evidence_units, 2, exact_fn)
        assert result["evidence_unit_recall"] == 1.0
