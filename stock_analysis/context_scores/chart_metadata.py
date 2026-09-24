CHART_PATTERN_META = {

    # ========================
    # REVERSAL PATTERNS
    # ========================
    "DoubleTopChartPattern": {
        "pattern_role": "reversal",
        "default_direction": "bearish",
        "include_in_score": True,
        "base_weight": 1.0,
    },

    "DoubleBottomChartPattern": {
        "pattern_role": "reversal",
        "default_direction": "bullish",
        "include_in_score": True,
        "base_weight": 1.0,
    },

    "TripleTopChartPattern": {
        "pattern_role": "reversal",
        "default_direction": "bearish",
        "include_in_score": True,
        "base_weight": 1.0,
    },

    "TripleBottomChartPattern": {
        "pattern_role": "reversal",
        "default_direction": "bullish",
        "include_in_score": True,
        "base_weight": 1.0,
    },

    "HeadAndShouldersChartPattern": {
        "pattern_role": "reversal",
        "default_direction": "bearish",
        "include_in_score": True,
        "base_weight": 1.0,
    },

    "InverseHeadAndShouldersChartPattern": {
        "pattern_role": "reversal",
        "default_direction": "bullish",
        "include_in_score": True,
        "base_weight": 1.0,
    },

    "DiamondTopChartPattern": {
        "pattern_role": "reversal",
        "default_direction": "bearish",
        "include_in_score": True,
        "base_weight": 1.0,
    },

    "DiamondBottomChartPattern": {
        "pattern_role": "reversal",
        "default_direction": "bullish",
        "include_in_score": True,
        "base_weight": 1.0,
    },

    "RisingWedgeChartPattern": {
        "pattern_role": "reversal",
        "default_direction": "bearish",
        "include_in_score": True,
        "base_weight": 1.0,
    },

    "FallingWedgeChartPattern": {
        "pattern_role": "reversal",
        "default_direction": "bullish",
        "include_in_score": True,
        "base_weight": 1.0,
    },


    # ========================
    # CONTINUATION PATTERNS
    # ========================
    "FlagChartPattern": {
        "pattern_role": "continuation",
        "default_direction": None,
        "include_in_score": True,
        "base_weight": 1.0,
    },

    "PennantChartPattern": {
        "pattern_role": "continuation",
        "default_direction": None,
        "include_in_score": True,
        "base_weight": 1.0,
    },

    "CupAndHandleChartPattern": {
        "pattern_role": "continuation",
        "default_direction": "bullish",
        "include_in_score": True,
        "base_weight": 1.0,
    },

    "AscendingTriangleChartPattern": {
        "pattern_role": "continuation",
        "default_direction": "bullish",
        "include_in_score": True,
        "base_weight": 1.0,
    },

    "DescendingTriangleChartPattern": {
        "pattern_role": "continuation",
        "default_direction": "bearish",
        "include_in_score": True,
        "base_weight": 1.0,
    },

    "AscendingChannelChartPattern": {
        "pattern_role": "continuation",
        "default_direction": "bullish",
        "include_in_score": True,
        "base_weight": 1.0,
    },

    "DescendingChannelChartPattern": {
        "pattern_role": "continuation",
        "default_direction": "bearish",
        "include_in_score": True,
        "base_weight": 1.0,
    },


    # ========================
    # NEUTRAL PATTERNS
    # ========================
    "SymmetricalTriangleChartPattern": {
        "pattern_role": "neutral",
        "default_direction": None,
        "include_in_score": False,
        "base_weight": 1.0,
    },

    "RectangleChartPattern": {
        "pattern_role": "neutral",
        "default_direction": None,
        "include_in_score": False,
        "base_weight": 1.0,
    },

    "MegaphoneChartPattern": {
        "pattern_role": "neutral",
        "default_direction": None,
        "include_in_score": False,
        "base_weight": 1.0,
    },
}