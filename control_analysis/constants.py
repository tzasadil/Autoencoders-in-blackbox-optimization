TABLE_DEFAULT_COLUMN_FORMAT = "|lc|"
BASELINE_COLOR = "#E04836"
DEFAULT_COLOR = "forestgreen"

FUNC_GROUP_LABELS = [
    ("f1-f5", 1, 5, "Separable Functions"),
    ("f6-f9", 6, 9, "Functions with low or moderate conditioning"),
    ("f10-f14", 10, 14, "Ill conditioned functions"),
    ("f15-f19", 15, 19, "Adequately structured multimodal functions"),
    ("f20-f24", 20, 24, "Weakly structured multimodal functions"),
]

EVAL_WINDOW_FUNC_GROUPS = [
    (1, 5, "Separable Functions"),
    (6, 9, "Functions with low or moderate conditioning"),
    (10, 14, "Ill conditioned functions"),
    (15, 19, "Adequately structured multimodal functions"),
    (20, 24, "Weakly structured multimodal functions"),
    (1, 24, "All functions"),
]