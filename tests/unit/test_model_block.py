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

    def test_any_one_condition_triggers(self):
        # LLM.md documents multiple conditions in one `when { }` block as
        # "any one triggers" (OR), not "all must hold" (AND) — either
        # condition alone must be enough to upgrade.
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
        # Neither condition true → no upgrade.
        assert r.select(context={"step": "other", "input_tokens": 5000}) == "claude-haiku"
        # Only step_is true → upgrades (OR, not AND).
        assert r.select(context={"step": "finalize", "input_tokens": 5000}) == "claude-opus"
        # Only tokens_gt true → upgrades.
        assert r.select(context={"step": "other", "input_tokens": 20000}) == "claude-opus"
        # Both true → still upgrades.
        assert r.select(context={"step": "finalize", "input_tokens": 20000}) == "claude-opus"

    def test_empty_conditions_never_fires(self):
        # An upgrade rule with no conditions must not be treated as "always
        # true" — that would upgrade unconditionally with no visible trigger.
        r = ModelRouter(
            default="claude-haiku",
            upgrades=[{"target": "claude-opus", "conditions": []}],
        )
        assert r.select(context={}) == "claude-haiku"

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


class TestCurrentStepScopingAtRuntime:
    """TestRouterUpgradeAtRuntime tests ModelRouter.select() in isolation
    with a hand-built context dict — it never exercises what actually
    populates context["step"] at runtime: step_decorator's _current_step
    tracking. That's where the real bug was: _current_step is set
    unconditionally on step entry but was never restored on exit, so a
    step that calls another step internally left _current_step stuck on
    the NESTED step's name even after that call returned — a `step is
    outer` upgrade rule would silently stop matching for any intent call
    the outer step makes after its nested call returns.
    """

    @pytest.mark.asyncio
    async def test_upgrade_rule_matches_after_nested_step_call_returns(
        self, transpile, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("DRIFT_USE_MOCK", "1")
        src = (
            'agent A { '
            '  model { '
            '    default: "claude-haiku" '
            '    upgrade to "claude-sonnet" when { step is outer } '
            '  } '
            '  step outer() -> string { '
            '    let inner_result = inner() '
            '    let after = classify "y" as string '
            '    return after '
            '  } '
            '  step inner() -> string { '
            '    return "done" '
            '  } '
            '}'
        )
        py = transpile(src).replace("Source: <drift_file>", "Source: inline")
        path = tmp_path / "gen_current_step.py"
        path.write_text(py)
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("gen_current_step", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["gen_current_step"] = mod
        spec.loader.exec_module(mod)

        agent = mod.A()
        selected_steps = []
        orig_select = agent.model.select
        def spy_select(remaining, context=None):
            selected_steps.append(context.get("step") if context else None)
            return orig_select(remaining, context=context)
        agent.model.select = spy_select

        await agent.outer()
        # The classify call happens AFTER inner() returns — it must see
        # _current_step == "outer" (not stuck on "inner" from the nested
        # call), so the upgrade rule actually fires.
        assert selected_steps == ["outer"]

    @pytest.mark.asyncio
    async def test_current_step_restored_to_none_after_top_level_call(
        self, transpile, tmp_path
    ):
        src = (
            'agent A { model: "claude-haiku" '
            '  step outer() -> string { '
            '    let inner_result = inner() '
            '    return inner_result '
            '  } '
            '  step inner() -> string { return "done" } '
            '}'
        )
        py = transpile(src).replace("Source: <drift_file>", "Source: inline")
        path = tmp_path / "gen_current_step2.py"
        path.write_text(py)
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("gen_current_step2", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["gen_current_step2"] = mod
        spec.loader.exec_module(mod)

        agent = mod.A()
        assert getattr(agent, "_current_step", None) is None
        await agent.outer()
        # Restored to the pre-call state (None), not stuck on "inner".
        assert getattr(agent, "_current_step", None) is None

    @pytest.mark.asyncio
    async def test_unavailability_mark_survives_a_nested_step_call(
        self, transpile, tmp_path
    ):
        # Third instance of the same scoping bug class as _drift_silent and
        # _current_step: step_decorator unconditionally called
        # self.model.reset_availability() on every step entry, including
        # NESTED ones. If the outer step marks a model unavailable (after a
        # ModelUnavailable error) and then calls another step internally
        # before its own retry/fallback logic is done relying on that mark,
        # the nested step's entry silently wiped it — so the outer step's
        # own subsequent model selection could pick the same broken model
        # right back up once the nested call returned.
        from drift.runtime import ModelUnavailable

        src = (
            'agent A { '
            '  model { default: "claude-haiku" fallback: "gpt-4o" } '
            '  step outer() -> string { '
            '    let x = classify "first" as string '
            '    let inner_result = inner() '
            '    let after = classify "y" as string '
            '    return after '
            '  } '
            '  step inner() -> string { '
            '    return "done" '
            '  } '
            '}'
        )
        py = transpile(src).replace("Source: <drift_file>", "Source: inline")
        path = tmp_path / "gen_unavail_scope.py"
        path.write_text(py)
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("gen_unavail_scope", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["gen_unavail_scope"] = mod
        spec.loader.exec_module(mod)

        agent = mod.A()
        calls = {"n": 0}

        # Patch one layer deeper than agent.intent itself: ModelUnavailable
        # retry now happens INSIDE intent() (see intent()'s docstring), so
        # replacing agent.intent wholesale would bypass that retry loop
        # entirely instead of exercising it. Patching the provider lets
        # the real intent()/mark_unavailable/retry path run for real,
        # which is what actually sets the unavailability mark this test
        # is checking survives a nested step call.
        class FlakyProvider:
            async def call(self, model, system, prompt, output_schema=None,
                            temperature=None):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise ModelUnavailable("down", model="claude-haiku")
                return ('"ok"', 10, 5)

        agent._provider_for = lambda model_name: FlakyProvider()

        seen_inside_inner = {}
        orig_inner = agent.inner

        async def spy_inner(*a, **kw):
            seen_inside_inner["unavailable"] = set(agent.model._unavailable)
            return await orig_inner(*a, **kw)

        agent.inner = spy_inner

        await agent.outer()
        # The mark made by outer's own fallback handling must still be
        # visible from inside the nested inner() call...
        assert seen_inside_inner["unavailable"] == {"claude-haiku"}
        # ...and must still be in effect after inner() returns and outer
        # resumes (not wiped by inner()'s own step entry).
        assert agent.model._unavailable == {"claude-haiku"}


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
