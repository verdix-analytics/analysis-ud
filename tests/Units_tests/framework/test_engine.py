import pandas as pd
import pytest
from pydantic import BaseModel

import stock_analysis.engine as engine
from stock_analysis.base import BasePattern


# Dummy result model helper

class DummyPatternResult:
    def __init__(self, confirmed=True, confidence=80.0, direction="neutral", extra=None):
        self.confirmed = confirmed
        self.confidence = confidence
        self.direction = direction
        self._extra = extra or {}

    def model_dump(self):
        base = {
            "confirmed": self.confirmed,
            "confidence": self.confidence,
            "direction": self.direction,
        }
        base.update(self._extra)
        return base


# Dummy pattern classes

class DummyConfirmedPattern(BasePattern):
    category = "chart"
    enabled = True

    def execute(self, df: pd.DataFrame, pre_trend=None):
        return DummyPatternResult(confirmed=True, confidence=85.0)


class DummyLowConfidencePattern(BasePattern):
    category = "chart"
    enabled = True

    def execute(self, df: pd.DataFrame, pre_trend=None):
        return DummyPatternResult(confirmed=True, confidence=20.0)


class DummyUnconfirmedPattern(BasePattern):
    category = "chart"
    enabled = True

    def execute(self, df: pd.DataFrame, pre_trend=None):
        return DummyPatternResult(confirmed=False, confidence=90.0)


class DummyExceptionPattern(BasePattern):
    category = "chart"
    enabled = True

    def execute(self, df: pd.DataFrame, pre_trend=None):
        raise RuntimeError("simulated pattern failure")


class DummySimplePattern(BasePattern):
    category = "candlestick"
    enabled = True

    def _run(self, df: pd.DataFrame) -> dict:
        return {"confirmed": True, "confidence": 75.0}


class DummyDisabledPattern(BasePattern):
    category = "chart"
    enabled = False

    def _run(self, df: pd.DataFrame) -> dict:
        return {"confirmed": True, "confidence": 75.0}



# Fixtures

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Datetime": pd.date_range("2024-01-01", periods=120, freq="D"),
        "Open": list(range(120)),
        "High": [x + 1 for x in range(120)],
        "Low": [x - 1 for x in range(120)],
        "Close": [x + 0.5 for x in range(120)],
    })


# Tests for _get_pattern_windows / _get_pretrend_window

def test_get_pattern_windows_success(monkeypatch):
    class DummyPattern(BasePattern):
        def _run(self, df): return {"confirmed": True, "confidence": 50}

    monkeypatch.setattr(engine.registry, "get", lambda name: 42)
    result = engine._get_pattern_windows(DummyPattern())
    assert result == 42


def test_get_pattern_windows_failure_returns_zero(monkeypatch):
    class DummyPattern(BasePattern):
        def _run(self, df): return {"confirmed": True, "confidence": 50}

    def raise_error(name):
        raise RuntimeError("registry failure")

    monkeypatch.setattr(engine.registry, "get", raise_error)
    result = engine._get_pattern_windows(DummyPattern())
    assert result == 0


def test_get_pretrend_window_success(monkeypatch):
    class DummyPattern(BasePattern):
        def _run(self, df): return {"confirmed": True, "confidence": 50}

    monkeypatch.setattr(engine.registry, "get_pretrend", lambda name, default: 15)
    result = engine._get_pretrend_window(DummyPattern())
    assert result == 15


def test_get_pretrend_window_failure_returns_default(monkeypatch):
    class DummyPattern(BasePattern):
        def _run(self, df): return {"confirmed": True, "confidence": 50}

    monkeypatch.setattr(engine.config, "PRE_TREND_WINDOW", 12, raising=False)

    def raise_error(name, default):
        raise RuntimeError("pretrend registry failure")

    monkeypatch.setattr(engine.registry, "get_pretrend", raise_error)
    result = engine._get_pretrend_window(DummyPattern())
    assert result == 12


# Tests for _single_pattern_exec

def test_single_pattern_exec_confirmed_returns_result(sample_df, monkeypatch):
    monkeypatch.setattr(
        engine.trend_analysis,
        "pre_trend_analysis",
        lambda df: {"label": "uptrend", "strength": 1.0}
    )
    monkeypatch.setattr(
        engine.trend_analysis,
        "post_trend_analysis",
        lambda df, signal: {"label": "downtrend", "strength": 0.5}
    )

    pre_df = sample_df.iloc[:10]
    post_df = sample_df.iloc[20:]

    result = engine._single_pattern_exec(
        sample_df.iloc[10:20].reset_index(drop=True),
        DummyConfirmedPattern(),
        start=10,
        end=20,
        pre_df=pre_df,
        post_df=post_df,
    )

    assert isinstance(result, dict)
    assert result["pattern"] == "DummyConfirmedPattern"
    assert result["category"] == "chart"
    assert result["window_start"] == 10
    assert result["window_end"] == 20
    assert result["confidence"] == 85.0
    assert result["confirmed"] is True
    assert result["signal"]["confirmed"] is True
    assert result["pre_trend"] == {"label": "uptrend", "strength": 1.0}
    assert result["post_trend"] == {"label": "downtrend", "strength": 0.5}


def test_single_pattern_exec_unconfirmed_returns_none(sample_df):
    result = engine._single_pattern_exec(sample_df, DummyUnconfirmedPattern())
    assert result is None


def test_single_pattern_exec_exception_returns_none(sample_df):
    result = engine._single_pattern_exec(sample_df, DummyExceptionPattern())
    assert result is None


def test_single_pattern_exec_no_pre_post_df_returns_none_for_trends(sample_df):
    result = engine._single_pattern_exec(
        sample_df.iloc[:20],
        DummyConfirmedPattern(),
        start=0,
        end=20,
        pre_df=None,
        post_df=None,
    )

    assert result["pre_trend"] is None
    assert result["post_trend"] is None


# Tests for _group_pattern_exec

def test_group_pattern_exec_collects_confirmed_results(sample_df):
    patterns = [DummyConfirmedPattern(), DummyUnconfirmedPattern()]
    result = engine._group_pattern_exec(
        window_df=sample_df.iloc[:20],
        patterns=patterns,
        window_count=3,
        pre_df=sample_df.iloc[:5],
        post_df=sample_df.iloc[20:30],
        max_workers=2,
    )

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["confirmed"] is True
    assert result[0]["window_count"] == 3


def test_group_pattern_exec_handles_pattern_failures(sample_df):
    patterns = [DummyConfirmedPattern(), DummyExceptionPattern()]
    result = engine._group_pattern_exec(
        window_df=sample_df.iloc[:20],
        patterns=patterns,
        window_count=1,
        pre_df=sample_df.iloc[:5],
        post_df=sample_df.iloc[20:30],
        max_workers=2,
    )

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["pattern"] == "DummyConfirmedPattern"


def test_group_pattern_exec_empty_patterns_returns_empty(sample_df):
    result = engine._group_pattern_exec(
        window_df=sample_df.iloc[:20],
        patterns=[],
        window_count=1,
        pre_df=sample_df.iloc[:5],
        post_df=sample_df.iloc[20:30],
        max_workers=2,
    )

    assert result == []


# Tests for get_patterns

def test_get_patterns_loads_enabled_pattern(monkeypatch):
    class FakeModule:
        MyPattern = DummySimplePattern
        Disabled = DummyDisabledPattern
        NotAPattern = object

    class FakePkg:
        __name__ = "fake_pkg"
        __path__ = ["fake_path"]

    monkeypatch.setattr(engine.pkgutil, "iter_modules", lambda path: [(None, "module1", None)])
    monkeypatch.setattr(engine.importlib, "import_module", lambda name: FakeModule)

    result = engine.get_patterns(FakePkg)

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], DummySimplePattern)


def test_get_patterns_handles_import_error(monkeypatch):
    class FakePkg:
        __name__ = "fake_pkg"
        __path__ = ["fake_path"]

    monkeypatch.setattr(engine.pkgutil, "iter_modules", lambda path: [(None, "module1", None)])

    def raise_import_error(name):
        raise RuntimeError("import failed")

    monkeypatch.setattr(engine.importlib, "import_module", raise_import_error)

    result = engine.get_patterns(FakePkg)
    assert result == []


# Tests for execute_patterns

def test_execute_patterns_collects_only_confidence_above_30(sample_df, monkeypatch):
    fake_formations = [
        {"confidence": 80, "confirmed": True, "pattern": "A"},
        {"confidence": 20, "confirmed": True, "pattern": "B"},
    ]

    monkeypatch.setattr(engine, "_group_pattern_exec", lambda *args, **kwargs: fake_formations)

    result = engine.execute_patterns(
        df=sample_df,
        window_size=20,
        patterns=[DummyConfirmedPattern()],
        limit_windows=2,
    )

    assert isinstance(result, list)
    assert all(item["confidence"] > 30 for item in result)
    assert len(result) > 0


def test_execute_patterns_respects_limit_windows(sample_df, monkeypatch):
    calls = []

    def fake_group(window_df, patterns, window_count, pre_df, post_df, max_workers=8):
        calls.append(window_count)
        return []

    monkeypatch.setattr(engine, "_group_pattern_exec", fake_group)

    engine.execute_patterns(
        df=sample_df,
        window_size=10,
        patterns=[DummyConfirmedPattern()],
        limit_windows=2,
    )

    # Because code uses <= limit_windows after increment logic,
    # it should run for window_count 1,2,3
    assert calls == [1, 2, 3]


def test_execute_patterns_empty_when_no_formations(sample_df, monkeypatch):
    monkeypatch.setattr(engine, "_group_pattern_exec", lambda *args, **kwargs: [])

    result = engine.execute_patterns(
        df=sample_df,
        window_size=20,
        patterns=[DummyConfirmedPattern()],
        limit_windows=2,
    )

    assert result == []


# Tests for category-specific exec functions

def test_chart_patterns_exec_returns_empty_if_not_enough_candles(sample_df, monkeypatch):
    monkeypatch.setattr(
        engine.config,
        "SLD_WINDOW_CONFIG",
        {
            "chart_patterns": {"min_candles": 200},
            "candlestick_patterns": {"min_candles": 5},
            "harmonic_patterns": {"min_candles": 5},
        },
        raising=False,
    )

    result = engine.chart_patterns_exec(sample_df, window_size=60)
    assert result == []


def test_chart_patterns_exec_calls_execute_patterns(sample_df, monkeypatch):
    monkeypatch.setattr(
        engine.config,
        "SLD_WINDOW_CONFIG",
        {
            "chart_patterns": {"min_candles": 10},
            "candlestick_patterns": {"min_candles": 5},
            "harmonic_patterns": {"min_candles": 5},
        },
        raising=False,
    )
    monkeypatch.setattr(engine, "CHART_PATTERNS", [DummyConfirmedPattern()])
    monkeypatch.setattr(engine, "execute_patterns", lambda df, window_size, patterns, limit_windows=5: [{"pattern": "X"}])

    result = engine.chart_patterns_exec(sample_df, window_size=60)
    assert result == [{"pattern": "X"}]


def test_candlestick_patterns_exec_returns_empty_if_not_enough_candles(sample_df, monkeypatch):
    monkeypatch.setattr(
        engine.config,
        "SLD_WINDOW_CONFIG",
        {
            "chart_patterns": {"min_candles": 5},
            "candlestick_patterns": {"min_candles": 200},
            "harmonic_patterns": {"min_candles": 5},
        },
        raising=False,
    )

    result = engine.candlestick_patterns_exec(sample_df, window_size=10)
    assert result == []


def test_candlestick_patterns_exec_calls_execute_patterns(sample_df, monkeypatch):
    monkeypatch.setattr(
        engine.config,
        "SLD_WINDOW_CONFIG",
        {
            "chart_patterns": {"min_candles": 5},
            "candlestick_patterns": {"min_candles": 10},
            "harmonic_patterns": {"min_candles": 5},
        },
        raising=False,
    )
    monkeypatch.setattr(engine, "CANDLESTICK_PATTERNS", [DummyConfirmedPattern()])
    monkeypatch.setattr(engine, "execute_patterns", lambda df, window_size, patterns, limit_windows=5: [{"pattern": "Y"}])

    result = engine.candlestick_patterns_exec(sample_df, window_size=10)
    assert result == [{"pattern": "Y"}]


def test_harmonic_patterns_exec_returns_empty_if_not_enough_candles(sample_df, monkeypatch):
    monkeypatch.setattr(
        engine.config,
        "SLD_WINDOW_CONFIG",
        {
            "chart_patterns": {"min_candles": 5},
            "candlestick_patterns": {"min_candles": 5},
            "harmonic_patterns": {"min_candles": 200},
        },
        raising=False,
    )

    result = engine.harmonic_patterns_exec(sample_df, window_size=120)
    assert result == []


def test_harmonic_patterns_exec_calls_execute_patterns(sample_df, monkeypatch):
    monkeypatch.setattr(
        engine.config,
        "SLD_WINDOW_CONFIG",
        {
            "chart_patterns": {"min_candles": 5},
            "candlestick_patterns": {"min_candles": 5},
            "harmonic_patterns": {"min_candles": 10},
        },
        raising=False,
    )
    monkeypatch.setattr(engine, "HARMONIC_PATTERNS", [DummyConfirmedPattern()])
    monkeypatch.setattr(engine, "execute_patterns", lambda df, window_size, patterns, limit_windows=5: [{"pattern": "Z", "confidence": 88}])

    result = engine.harmonic_patterns_exec(sample_df, window_size=120)
    assert result == [{"pattern": "Z", "confidence": 88}]


# Tests for exec_all_patterns

def test_exec_all_patterns_success(sample_df, monkeypatch):
    monkeypatch.setattr(engine, "chart_patterns_exec", lambda df: [{"pattern": "chart"}])
    monkeypatch.setattr(engine, "candlestick_patterns_exec", lambda df: [{"pattern": "candle"}])
    monkeypatch.setattr(engine, "harmonic_patterns_exec", lambda df: [{"pattern": "harmonic"}])

    class DummyFuture:
        def __init__(self, result):
            self._result = result
        def result(self):
            return self._result

    class DummyExecutor:
        def __init__(self, max_workers=None): pass
        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): pass
        def submit(self, fn, df):
            return DummyFuture(fn(df))

    monkeypatch.setattr(engine, "ProcessPoolExecutor", DummyExecutor)
    monkeypatch.setattr(engine, "as_completed", lambda futures: list(futures.keys()))

    result = engine.exec_all_patterns(sample_df)

    assert result["chart_patterns"] == [{"pattern": "chart"}]
    assert result["candlestick_patterns"] == [{"pattern": "candle"}]
    assert result["harmonic_patterns"] == [{"pattern": "harmonic"}]


def test_exec_all_patterns_fallback_on_executor_failure(sample_df, monkeypatch):
    monkeypatch.setattr(engine, "chart_patterns_exec", lambda df: [{"pattern": "chart"}])
    monkeypatch.setattr(engine, "candlestick_patterns_exec", lambda df: [{"pattern": "candle"}])
    monkeypatch.setattr(engine, "harmonic_patterns_exec", lambda df: [{"pattern": "harmonic"}])

    class BrokenExecutor:
        def __init__(self, max_workers=None): pass
        def __enter__(self): raise RuntimeError("executor failed")
        def __exit__(self, exc_type, exc, tb): pass

    monkeypatch.setattr(engine, "ProcessPoolExecutor", BrokenExecutor)

    result = engine.exec_all_patterns(sample_df)

    assert result["chart_patterns"] == [{"pattern": "chart"}]
    assert result["candlestick_patterns"] == [{"pattern": "candle"}]
    assert result["harmonic_patterns"] == [{"pattern": "harmonic"}]


def test_exec_all_patterns_fallback_handles_inner_failure(sample_df, monkeypatch):
    monkeypatch.setattr(engine, "chart_patterns_exec", lambda df: [{"pattern": "chart"}])
    monkeypatch.setattr(engine, "candlestick_patterns_exec", lambda df: (_ for _ in ()).throw(RuntimeError("inner fail")))
    monkeypatch.setattr(engine, "harmonic_patterns_exec", lambda df: [{"pattern": "harmonic"}])

    class BrokenExecutor:
        def __init__(self, max_workers=None): pass
        def __enter__(self): raise RuntimeError("executor failed")
        def __exit__(self, exc_type, exc, tb): pass

    monkeypatch.setattr(engine, "ProcessPoolExecutor", BrokenExecutor)

    result = engine.exec_all_patterns(sample_df)

    assert result["chart_patterns"] == [{"pattern": "chart"}]
    assert result["candlestick_patterns"] == []
    assert result["harmonic_patterns"] == [{"pattern": "harmonic"}]


# Tests for exec_pattern

def test_exec_pattern_start_less_equal_zero_returns_none(sample_df):
    pattern = DummyConfirmedPattern()
    result = engine.exec_pattern(sample_df, pattern, start=0)
    assert result is None


def test_exec_pattern_end_out_of_range_returns_none(sample_df, monkeypatch):
    pattern = DummyConfirmedPattern()
    monkeypatch.setattr(engine, "_get_pattern_windows", lambda pattern: 1000)

    result = engine.exec_pattern(sample_df, pattern, start=10)
    assert result is None


def test_exec_pattern_calls_single_pattern_exec(sample_df, monkeypatch):
    pattern = DummyConfirmedPattern()
    monkeypatch.setattr(engine, "_get_pattern_windows", lambda pattern: 10)

    called = {}

    def fake_single(df, pattern, start, end, pre_df, post_df):
        called["args"] = (start, end)
        return {"pattern": "DummyConfirmedPattern"}

    monkeypatch.setattr(engine, "_single_pattern_exec", fake_single)

    engine.exec_pattern(sample_df, pattern, start=5)

    assert called["args"] == (5, 15)