from mva.service.query_understanding import (
    RuleBasedConstraintParser, LLMConstraintParser, HybridConstraintParser,
)


class _FakeLLM:
    """有 .complete() 的假云端 LLM。"""
    def __init__(self, reply): self.reply = reply; self.calls = 0
    def complete(self, prompt, max_new_tokens=200):
        self.calls += 1
        return self.reply


def test_llm_parser_parses_json_with_fences():
    llm = _FakeLLM('```json\n{"view": 2, "time_start": null, '
                   '"time_end": null, "semantic_text": "白色SUV"}\n```')
    c = LLMConstraintParser(llm).parse("水库那个无人机里的白色SUV")
    assert c.view_ref == "2"
    assert c.semantic_text == "白色SUV"
    assert c.source == "llm"


def test_llm_parser_garbage_degrades():
    c = LLMConstraintParser(_FakeLLM("抱歉我无法理解")).parse("随便一句")
    assert c.view_ref is None
    assert c.semantic_text == "随便一句"
    assert c.source == "none"


def test_hybrid_rule_hit_skips_llm():
    llm = _FakeLLM('{"view":9,"time_start":null,"time_end":null,"semantic_text":"x"}')
    h = HybridConstraintParser(RuleBasedConstraintParser(), LLMConstraintParser(llm))
    c = h.parse("视角1里的黄车")
    assert c.view_ref == "1"          # 规则命中
    assert llm.calls == 0             # 不调 LLM


def test_hybrid_rule_miss_with_trigger_calls_llm():
    llm = _FakeLLM('{"view":3,"time_start":null,"time_end":null,"semantic_text":"人"}')
    h = HybridConstraintParser(RuleBasedConstraintParser(), LLMConstraintParser(llm))
    c = h.parse("水库那个无人机里的人")     # 规则抓不到数字, 但含触发词"无人机"
    assert llm.calls == 1
    assert c.view_ref == "3"


def test_hybrid_rule_miss_no_trigger_skips_llm():
    llm = _FakeLLM('{"view":3,"time_start":null,"time_end":null,"semantic_text":"人"}')
    h = HybridConstraintParser(RuleBasedConstraintParser(), LLMConstraintParser(llm))
    c = h.parse("黄色的车")               # 无触发词
    assert llm.calls == 0
    assert c.source == "none"


def test_hybrid_no_llm_ok():
    h = HybridConstraintParser(RuleBasedConstraintParser(), None)
    c = h.parse("水库那个无人机里的人")
    assert c.source == "none"          # 没 llm 就退回规则空结果
