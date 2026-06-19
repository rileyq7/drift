"""Tests for §4 model block form with `upgrade to ... when ...` rules."""
import pytest

from drift import ast_nodes as ast
from drift.runtime import ModelRouter, Confident


def get_model(src: str, parse_ast):
    return parse_ast(src).declarations[0].model_config


class TestModelBlockParse:
    BLOCK = (
        'agent A { model { '
        '  default: "claude-haiku" '
        '  upgrade to "claude-sonnet" when { confidence < 0.8 input_tokens > 10000 } '
        '  upgrade to "claude-opus" when { step is finalize } '
        '  fallback: "gpt-4o", "gpt-4o-mini" '
        '  never: "gpt-3.5-turbo" '
        '} step f() { respond "x" } }'
    )

    def test_default(self, parse_ast):
        m = get_model(self.BLOCK, parse_ast)
        assert m.default == "claude-haiku"

    def test_fallback_list(self, parse_ast):
        m = get_model(self.BLOCK, parse_ast)
        assert m.fallback_list == ["gpt-4o", "gpt-4o-mini"]

    def test_never_list(self, parse_ast):
        m = get_model(self.BLOCK, parse_ast)
        assert m.never_list == ["gpt-3.5-turbo"]

    def test_two_upgrade_rules(self, parse_ast):
        m = get_model(self.BLOCK, parse_ast)
        assert len(m.upgrades) == 2
        assert m.upgrades[0].target_model == "claude-sonnet"
        assert m.upgrades[1].target_model == "claude-opus"

    def test_multi_condition_rule(self, parse_ast):
        m = get_model(self.BLOCK, parse_ast)
        conds = m.upgrades[0].conditions
        assert len(conds) == 2
        assert {c.kind for c in conds} == {"confidence_lt", "tokens_gt"}

    def test_step_is_condition(self, parse_ast):
        m = get_model(self.BLOCK, parse_ast)
        cond = m.upgrades[1].conditions[0]
        assert cond.kind == "step_is"
        assert cond.value == "finalize"


class TestModelBlockCodegen:
    def test_upgrades_in_router_init(self, transpile):
        out = transpile(
            'agent A { model { '
            '  default: "claude-haiku" '
            '  upgrade to "claude-opus" when { step is finalize } '
            '} step f() { respond "x" } }'
        )
        assert "upgrades=" in out
        assert '"target": "claude-opus"' in out
        assert '"kind": "step_is"' in out


class TestRouterUpgradeAtRuntime:
    def test_step_is_upgrade_fires(self):
        r = ModelRouter(
            default="claude-haiku",
            upgrades=[{
                "target": "claude-opus",
                "conditions": [{"kind": "step_is", "value": "finalize"}],
            }],
        )
        assert r.select(context={"step": "finalize"}) == "claude-opus"
        # Wrong step → no upgrade
        assert r.select(context={"step": "draft"}) == "claude-haiku"

    def test_tokens_gt_upgrade_fires(self):
        r = ModelRouter(
            default="claude-haiku",
            upgrades=[{
                "target": "claude-sonnet",
                "conditions": [{"kind": "tokens_gt", "value": 10000}],
            }],
        )
        assert r.select(context={"input_tokens": 20000}) == "claude-sonnet"
        assert r.select(context={"input_tokens": 5000}) == "claude-haiku"

    def test_all_conditions_must_match(self):
        # Both conditions present, only one true → no upgrade.
        r = ModelRouter(
            default="claude-haiku",
            upgrades=[{
                "target": "claude-opus",
                "conditions": [
                    {"kind": "step_is", "value": "finalize"},
                    {"kind": "tokens_gt", "value": 10000},
                ],
            }],
        )
        assert r.select(context={"step": "finalize", "input_tokens": 5000}) == "claude-haiku"
        assert r.select(context={"step": "finalize", "input_tokens": 20000}) == "claude-opus"

    def test_confidence_lt_uses_last_recorded(self):
        r = ModelRouter(
            default="claude-haiku",
            upgrades=[{
                "target": "claude-sonnet",
                "conditions": [{"kind": "confidence_lt", "value": 0.8}],
            }],
        )
        # No confidence yet → default 1.0, doesn't satisfy <0.8
        assert r.select(context={}) == "claude-haiku"
        r.record_confidence(0.5)
        assert r.select(context={}) == "claude-sonnet"
        # Restored confidence → no upgrade
        r.record_confidence(0.95)
        assert r.select(context={}) == "claude-haiku"

    def test_upgrade_skipped_when_target_is_unavailable(self):
        r = ModelRouter(
            default="claude-haiku",
            upgrades=[{
                "target": "claude-opus",
                "conditions": [{"kind": "step_is", "value": "finalize"}],
            }],
        )
        r.mark_unavailable("claude-opus")
        assert r.select(context={"step": "finalize"}) == "claude-haiku"

    def test_first_matching_upgrade_wins(self):
        # Two rules, both conditions true; first one wins.
        r = ModelRouter(
            default="claude-haiku",
            upgrades=[
                {"target": "claude-sonnet",
                 "conditions": [{"kind": "step_is", "value": "x"}]},
                {"target": "claude-opus",
                 "conditions": [{"kind": "step_is", "value": "x"}]},
            ],
        )
        assert r.select(context={"step": "x"}) == "claude-sonnet"


class TestColonFormStillWorks:
    """The block form is additive — the simple `model: "x"` syntax must still parse."""

    def test_simple_colon_form(self, parse_ast):
        d = parse_ast(
            'agent A { model: "claude-sonnet" step f() { respond "x" } }'
        ).declarations[0]
        assert d.model_config.default == "claude-sonnet"

    def test_prefer_fallback_colon_form(self, parse_ast):
        d = parse_ast(
            'agent A { model: prefer "claude-sonnet" fallback "gpt-4o" '
            'step f() { respond "x" } }'
        ).declarations[0]
        assert d.model_config.prefer == "claude-sonnet"
        assert d.model_config.fallback == "gpt-4o"
