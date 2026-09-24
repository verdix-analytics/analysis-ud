import importlib
import pkgutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import sys
import pandas as pd
import logging
from itertools import groupby
import numpy as np
from typing import cast

import stock_analysis.chart_patterns as chart_pg
import stock_analysis.candlestick_patterns as candlestick_pg
import stock_analysis.harmonic_patterns as harmonic_pg
from stock_analysis.base import BasePattern
import stock_analysis.config as config
from Utilities.window_registry import registry
from stock_analysis.trendanalysis import TrendAnalysis
from .technical_analysis import TechnicalAnalysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

trend_analysis = TrendAnalysis()
ta_engine = TechnicalAnalysis()

def _get_pattern_windows(pattern: BasePattern) -> int:
    try:
        return registry.get(pattern.__class__.__name__)
    except:
        logger.error(f"Failure to load the window size for the pattern: {pattern.__class__.__name__}")
    return 0

def _get_pretrend_window(pattern: BasePattern) -> int:
    try:
        return registry.get_pretrend(
            pattern.__class__.__name__,
            default=config.PRE_TREND_WINDOW,
        )
    except Exception:
        logger.error(
            "Failure to load the pre-trend window for the pattern: %s",
            pattern.__class__.__name__,
        )
    return config.PRE_TREND_WINDOW

def _single_pattern_exec(df: pd.DataFrame, 
                         pattern: BasePattern, 
                         start: int | None = None, 
                         end: int | None = None,
                         pre_df: pd.DataFrame | None = None,
                         post_df: pd.DataFrame | None = None) -> dict | None:
    try:
        result = pattern.execute(df)
        signal = result.model_dump()
        
        if not result.confirmed:
            return None

        return {
            'pattern': pattern.__class__.__name__,
            'category': pattern.category,
            'window_start': start,
            'window_end': end,
            'confidence': result.confidence,
            'confirmed': result.confirmed,
            'signal': signal,
            'pre_trend': trend_analysis.pre_trend_analysis(pre_df) if pre_df is not None and len(pre_df) > 0 else None,
            'post_trend': trend_analysis.post_trend_analysis(post_df, signal) if post_df is not None and len(post_df) > 0 else None
        }
    except Exception as e:
        logger.error(f"Pattern {pattern.__class__.__name__} failed: {e}")
        return None

def _group_pattern_exec(window_df: pd.DataFrame, 
                        patterns: list, 
                        window_count: int,
                        pre_df: pd.DataFrame,
                        post_df: pd.DataFrame,
                        max_workers: int = 8) -> list:
    results = []
    try:
        with ThreadPoolExecutor(max_workers = max_workers) as executor:
            futures = {
                executor.submit(
                    _single_pattern_exec,
                    window_df, pattern, None, None, pre_df, post_df
                ): pattern
                for pattern in patterns
            }

            for future in as_completed(futures):
                result = future.result()
                if result and result.__contains__("confirmed") and result["confirmed"] == True:
                    result["window_count"] = window_count #we count window starting from the right (right -> left: window_count + 1)
                    results.append(result)
    except Exception as e:
        logger.error(f"Unable to execute the list of patterns. \n Reason of failure: {e}")
    return results
        
def get_patterns(pkg) -> list:
    patterns = []
    try:
        logger.info(f"Loading the algorithms for {pkg.__name__}")
        for _, module_name, _ in pkgutil.iter_modules(pkg.__path__):
            module = importlib.import_module(f"{pkg.__name__}.{module_name}")
            for attribute in dir(module):
                attr = getattr(module, attribute)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, BasePattern)
                    and attr is not BasePattern
                    and attr.enabled
                ):
                    patterns.append(attr())
                    logger.info(f"Got the following pattern: {attr.__name__}")
    except Exception as e:
        logger.error(f"Unable to load pattern detection algorithms due to : {e}")
        logger.info(f"Unable to get all patterns - total number of patterns loaded : {len(patterns)}/40")
    return patterns

CHART_PATTERNS = get_patterns(chart_pg)
CANDLESTICK_PATTERNS = get_patterns(candlestick_pg)
HARMONIC_PATTERNS = get_patterns(harmonic_pg)

def technical_indicators(df: pd.DataFrame):
    if df is None or len(df) < 35:
        return 0.5
    
    indicators = ta_engine.latest_snapshot(df)
    p_di = indicators.get("plus_di") or 50
    m_di = indicators.get("minus_di") or 50
    directional_bias = (p_di - m_di) / (p_di + m_di) if (p_di + m_di) > 0 else 0
    
    rsi_val = indicators.get("rsi") or 50
    s_rsi = (rsi_val - 50) / 50
    
    macd_hist = indicators.get("macd_hist") or 0
    s_macd = np.tanh(macd_hist * 10)
    
    adx_weight = (indicators.get("adx") or 0) / 100
    raw_mss = (directional_bias * adx_weight) + (((s_rsi + s_macd)/2) * (1-adx_weight))
    
    indicators_dict = {
        "Positive Bias": p_di,
        "Negative Bias": m_di,
        "RSI": s_rsi,
        "MACD": s_macd,
        "ADX Weight": adx_weight,
        "Market Sentiment": float((raw_mss+1)/2),
    }
    return indicators_dict
    
def execute_patterns(df: pd.DataFrame, 
                     window_size: int, 
                     patterns: list,
                     limit_windows: int = sys.maxsize-1) -> list:
    final_result = []
    window_count = 0
    ta_result = technical_indicators(df)
    for start_idx in range(len(df)-window_size, -1, -window_size//2):
        if window_count <= limit_windows:
            end_idx = start_idx + window_size
            window_df: pd.DataFrame = cast(pd.DataFrame, df.iloc[start_idx:end_idx].reset_index(drop=True))
            pre_df: pd.DataFrame = cast(pd.DataFrame, df.iloc[:start_idx].reset_index(drop=True))
            post_df: pd.DataFrame = cast(pd.DataFrame, df.iloc[end_idx:].reset_index(drop=True))
            window_count += 1
            formations = _group_pattern_exec(window_df, 
                                            patterns, 
                                            window_count,
                                            pre_df,
                                            post_df)
            
            for formation in formations:
                formation["technical_analysis"] = ta_result
                if formation['confidence'] > 30:
                    final_result.append(formation)
    return final_result

def chart_patterns_exec(df: pd.DataFrame, window_size: int = 60) -> list[dict]:
    logger.info("Running chart patterns")
    chart_config = config.SLD_WINDOW_CONFIG['chart_patterns']

    if len(df) < chart_config['min_candles']:
        logger.warning(
            f"Not enough candles for chart patterns: "
            f"{len(df)} < {chart_config['min_candles']}"
        )
        return []

    final_result = execute_patterns(df, window_size, CHART_PATTERNS)
    logger.info("Completed Chart pattern identifications")
    for res in final_result:
        logger.info(f"Pattern Found: {res['pattern']} - confidence: {res['confidence']}")
    return final_result

def candlestick_patterns_exec(df: pd.DataFrame, window_size=5) -> list[dict]:
    logger.info("Running candlestick patterns")
    candlestick_config = config.SLD_WINDOW_CONFIG['candlestick_patterns']

    if len(df) < candlestick_config['min_candles']:
        logger.warning(
            f"Not enough candles for candlestick patterns: "
            f"{len(df)} < {candlestick_config['min_candles']}"
        )
        return []

    final_result = execute_patterns(df, window_size, CANDLESTICK_PATTERNS)
    logger.info("Completed Candlestick pattern identifications")
    for res in final_result:
        logger.info(f"Pattern Found: {res['pattern']} - confidence: {res['confidence']}")
    return final_result

def harmonic_patterns_exec(df: pd.DataFrame, window_size = 120) -> list[dict]:
    logger.info("Running harmonic patterns")
    harmonic_config = config.SLD_WINDOW_CONFIG['harmonic_patterns']

    if len(df) < harmonic_config['min_candles']:
        logger.warning(
            f"Not enough candles for harmonic patterns: "
            f"{len(df)} < {harmonic_config['min_candles']}"
        )
        return []

    final_result = execute_patterns(df, window_size, HARMONIC_PATTERNS)
    logger.info("Completed harmonic pattern identifications")
    for res in final_result:
        logger.info(f"Pattern Found: {res['pattern']} - confidence: {res['confidence']}")
    return final_result

def exec_all_patterns(df:pd.DataFrame) -> dict:
    results = {}
    categories = {
        'chart_patterns': chart_patterns_exec,
        'candlestick_patterns': candlestick_patterns_exec,
        'harmonic_patterns': harmonic_patterns_exec,
    }
    try:
        with ProcessPoolExecutor(max_workers = 3) as executor:
            futures = {
                    executor.submit(fn, df): category
                    for category, fn in categories.items()
            }
            for future in as_completed(futures):
                category = futures[future]
                results[category] = future.result()
        logger.info("Successfully executed all the patterns for charts + candlestick ratios + harmonic patterns")
    except Exception as e:
        logger.error(f"Failure to execute all the patterns for charts + candlestick ratios + harmonic patterns. \n Reason of Failure: {e}")
        logger.info("Falling back to in-process execution for all pattern categories")
        for category, fn in categories.items():
            try:
                results[category] = fn(df)
            except Exception as inner_e:
                logger.error(f"Fallback execution failed for {category}: {inner_e}")
                results[category] = []
    return results

def exec_pattern(df:pd.DataFrame, pattern: BasePattern, start: int):
    if start <=0:
        logger.warning("Leave some data for pre-trend analysis")
        return
    ws = _get_pattern_windows(pattern)
    end = start + ws
    if end >= len(df):
        logger.warning("Leave some data for post-trend analysis")
        return
    _single_pattern_exec(df,pattern, start, end, None, None)