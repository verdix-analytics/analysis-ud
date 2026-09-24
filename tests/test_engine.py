# tests/test_engine.py
import pytest
import stock_analysis.config as config

from stock_analysis.engine import (
    exec_all_patterns,
    chart_patterns_exec,
    candlestick_patterns_exec,
    harmonic_patterns_exec,
    CHART_PATTERNS,
    CANDLESTICK_PATTERNS,
    HARMONIC_PATTERNS,
)


# ── Test 1 — pattern loading ───────────────────────────────────────────────────

class TestPatternLoading:

    def test_chart_patterns_loaded(self):
        assert len(CHART_PATTERNS) > 0, \
            "No chart patterns loaded — check stock_analysis/chart_patterns/"

    def test_candlestick_patterns_loaded(self):
        assert len(CANDLESTICK_PATTERNS) > 0, \
            "No candlestick patterns loaded — check stock_analysis/candlestick_patterns/"

    def test_harmonic_patterns_loaded(self):
        assert len(HARMONIC_PATTERNS) > 0, \
            "No harmonic patterns loaded — check stock_analysis/harmonic_patterns/"

    def test_all_patterns_have_enabled_attribute(self):
        all_patterns = CHART_PATTERNS + CANDLESTICK_PATTERNS + HARMONIC_PATTERNS
        for p in all_patterns:
            assert hasattr(p, 'enabled'), \
                f"{p.__class__.__name__} missing 'enabled'"

    def test_all_patterns_have_category_attribute(self):
        all_patterns = CHART_PATTERNS + CANDLESTICK_PATTERNS + HARMONIC_PATTERNS
        for p in all_patterns:
            assert hasattr(p, 'category'), \
                f"{p.__class__.__name__} missing 'category'"

    def test_all_patterns_have_execute_method(self):
        all_patterns = CHART_PATTERNS + CANDLESTICK_PATTERNS + HARMONIC_PATTERNS
        for p in all_patterns:
            assert hasattr(p, 'execute'), \
                f"{p.__class__.__name__} missing 'execute'"

    def test_all_patterns_have_run_method(self):
        all_patterns = CHART_PATTERNS + CANDLESTICK_PATTERNS + HARMONIC_PATTERNS
        for p in all_patterns:
            assert hasattr(p, '_run'), \
                f"{p.__class__.__name__} missing '_run'"

    def test_chart_patterns_have_correct_category(self):
        for p in CHART_PATTERNS:
            assert p.category == 'chart', \
                f"{p.__class__.__name__} has wrong category: {p.category}"

    def test_candlestick_patterns_have_correct_category(self):
        for p in CANDLESTICK_PATTERNS:
            assert p.category == 'candlestick', \
                f"{p.__class__.__name__} has wrong category: {p.category}"

    def test_harmonic_patterns_have_correct_category(self):
        for p in HARMONIC_PATTERNS:
            assert p.category == 'harmonic', \
                f"{p.__class__.__name__} has wrong category: {p.category}"

    def test_all_patterns_are_enabled(self):
        all_patterns = CHART_PATTERNS + CANDLESTICK_PATTERNS + HARMONIC_PATTERNS
        for p in all_patterns:
            assert p.enabled is True, \
                f"{p.__class__.__name__} is not enabled"


# ── Test 2 — exec_all_patterns returns all 3 categories ───────────────────────

class TestExecAllPatterns:

    def test_returns_dict(self, dummy_df_large):
        result = exec_all_patterns(dummy_df_large)
        assert isinstance(result, dict), \
            f"exec_all_patterns should return dict, got {type(result)}"

    def test_returns_chart_patterns_key(self, dummy_df_large):
        result = exec_all_patterns(dummy_df_large)
        assert 'chart_patterns' in result, \
            "exec_all_patterns missing 'chart_patterns' key"

    def test_returns_candlestick_patterns_key(self, dummy_df_large):
        result = exec_all_patterns(dummy_df_large)
        assert 'candlestick_patterns' in result, \
            "exec_all_patterns missing 'candlestick_patterns' key"

    def test_returns_harmonic_patterns_key(self, dummy_df_large):
        result = exec_all_patterns(dummy_df_large)
        assert 'harmonic_patterns' in result, \
            "exec_all_patterns missing 'harmonic_patterns' key"

    def test_each_category_is_list(self, dummy_df_large):
        result = exec_all_patterns(dummy_df_large)
        for category, formations in result.items():
            assert isinstance(formations, list), \
                f"{category} should be list, got {type(formations)}"

    def test_max_formations_per_category(self, dummy_df_large):
        result = exec_all_patterns(dummy_df_large)
        for category, formations in result.items():
            assert len(formations) <= config.NO_OF_PATTERNS, \
                f"{category} returned {len(formations)} formations, " \
                f"max is {config.NO_OF_PATTERNS}"

    def test_returns_empty_dict_on_empty_df(self):
        import pandas as pd
        empty_df = pd.DataFrame(
            columns=['open', 'high', 'low', 'close', 'volume']
        )
        result = exec_all_patterns(empty_df)
        assert isinstance(result, dict), \
            "Should return dict even for empty df"


# ── Test 3 — window position validity ─────────────────────────────────────────

class TestWindowPositions:

    def test_window_start_is_non_negative(self, dummy_df_large):
        result = exec_all_patterns(dummy_df_large)
        for category, formations in result.items():
            for f in formations:
                assert f['window_start'] >= 0, \
                    f"{category}/{f['pattern']} has negative window_start: {f['window_start']}"

    def test_window_end_within_df_length(self, dummy_df_large):
        result = exec_all_patterns(dummy_df_large)
        for category, formations in result.items():
            for f in formations:
                assert f['window_end'] <= len(dummy_df_large), \
                    f"{category}/{f['pattern']} window_end {f['window_end']} " \
                    f"exceeds df length {len(dummy_df_large)}"

    def test_window_start_less_than_window_end(self, dummy_df_large):
        result = exec_all_patterns(dummy_df_large)
        for category, formations in result.items():
            for f in formations:
                assert f['window_start'] < f['window_end'], \
                    f"{category}/{f['pattern']} window_start {f['window_start']} " \
                    f">= window_end {f['window_end']}"

    def test_window_start_has_enough_pre_trend_candles(self, dummy_df_large):
        result = exec_all_patterns(dummy_df_large)
        for category, formations in result.items():
            for f in formations:
                assert f['window_start'] >= config.PRE_TREND_WINDOW, \
                    f"{category}/{f['pattern']} window_start {f['window_start']} " \
                    f"too early — needs {config.PRE_TREND_WINDOW} pre-trend candles"

    def test_window_end_has_enough_post_trend_candles(self, dummy_df_large):
        result = exec_all_patterns(dummy_df_large)
        for category, formations in result.items():
            for f in formations:
                assert f['window_end'] + config.POST_TREND_WINDOW <= len(dummy_df_large), \
                    f"{category}/{f['pattern']} window_end {f['window_end']} " \
                    f"too late — needs {config.POST_TREND_WINDOW} post-trend candles"

    def test_window_start_and_end_are_integers(self, dummy_df_large):
        result = exec_all_patterns(dummy_df_large)
        for category, formations in result.items():
            for f in formations:
                assert isinstance(f['window_start'], int), \
                    f"{category}/{f['pattern']} window_start is not int"
                assert isinstance(f['window_end'], int), \
                    f"{category}/{f['pattern']} window_end is not int"

    def test_formations_sorted_by_most_recent(self, dummy_df_large):
        """Last n formations should be sorted most recent first."""
        result = exec_all_patterns(dummy_df_large)
        for category, formations in result.items():
            starts = [f['window_start'] for f in formations]
            assert starts == sorted(starts, reverse=True), \
                f"{category} formations not sorted by most recent window_start"

    def test_one_formation_per_window(self, dummy_df_large):
        """get_last_n_formations deduplicates by window — no two formations same window."""
        result = exec_all_patterns(dummy_df_large)
        for category, formations in result.items():
            starts = [f['window_start'] for f in formations]
            assert len(starts) == len(set(starts)), \
                f"{category} has duplicate window_start values — dedup not working"