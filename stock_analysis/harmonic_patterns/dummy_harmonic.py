from stock_analysis.base import BasePattern
import pandas as pd
from stock_analysis.Enums.category import Category

class DummyHarmonicPattern(BasePattern):
    enabled  = False
    category = Category.HARMONIC.value
    expected_pretrend = None

    def _run(self, df: pd.DataFrame) -> dict:
        return {
            'confirmed':  True,
            'confidence': 0.80,
        }
