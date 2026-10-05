"""
Configuration parameters for the IncrementalEngage streaming and uplift pipeline.
"""

CONFIG = {
    "app_name": "IncrementalEngageStream",
    "master_url": "local[*]",
    "data_path": "data/ecom_clickstream.csv",
    "treatment_col": "is_targeted",
    "outcome_col": "purchased",
    "feature_cols": [
        "scroll_velocity",
        "dwell_time_sec",
        "cart_item_count",
        "historical_spend",
        "session_depth"
    ],
    "test_size": 0.2,
    "random_state": 42
}