"""Settings for each environment. ``base`` applies to every profile (see config.load_profile)."""

PROFILES = {
    "base": {
        "bank_feed_url": "https://feed.example-bank.test/v2",
        "ledger_dsn": "postgresql://reconcile@localhost/ledger",
        "http_timeout": 10.0,
        "retries": {
            "default": {"max_retries": 3, "base_delay": 1.0, "factor": 2.0, "max_delay": 30.0},
        },
    },
    "dev": {
        "retries": {"default": {"max_retries": 0}},
    },
    "staging": {
        "ledger_dsn": "postgresql://reconcile@staging-db/ledger",
        "retries": {
            "default": {"max_retries": 3, "base_delay": 1.0, "factor": 2.0, "max_delay": 30.0},
            "bankfeed": {"max_retries": 2, "base_delay": 5.0, "factor": 2.0},
        },
    },
    "production": {
        "ledger_dsn": "postgresql://reconcile@prod-db/ledger",
        "http_timeout": 20.0,
        "retries": {
            "default": {"max_retries": 3, "base_delay": 1.0, "factor": 2.0, "max_delay": 30.0},
            "bankfeed": {"max_retries": 4, "base_delay": 1.5},
            "ledger": {"max_retries": 6, "base_delay": 0.2, "factor": 2.0, "max_delay": 5.0},
        },
    },
}
