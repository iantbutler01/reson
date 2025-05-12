BIG_MODEL = "BIG_MODEL"
MODEL = "MODEL"
SMALL_MODEL = "SMALL_MODEL"

MODEL_CONFIGURATION = {
    "openrouter": {
        "default": {
            "BIG_MODEL": "anthropic/claude-3.7-sonnet",
            "MODEL": "anthropic/claude-3.5-haiku",
            "SMALL_MODEL": "anthropic/claude-3-haiku",
        },
        "anthropic": {
            "BIG_MODEL": "anthropic/claude-3.7-sonnet",
            "MODEL": "anthropic/claude-3.5-haiku",
            "SMALL_MODEL": "anthropic/claude-3-haiku",
        },
        "openai": {
            "BIG_MODEL": "openai/o3",
            "MODEL": "openai/o4-mini",
            "SMALL_MODEL": "openai/o4-mini",
        },
    },
    "anthropic": {
        "default": {
            "BIG_MODEL": "claude-3-7-sonnet-20250219",
            "MODEL": "claude-3-5-haiku-20241022",
            "SMALL_MODEL": "claude-3-5-haiku-20241022",
        },
        "thinking": {
            "BIG_MODEL": "claude-3-7-sonnet-20250219",
            "MODEL": "claude-3-5-haiku-20241022",
            "SMALL_MODEL": "claude-3-5-haiku-20241022",
        },
    },
    "bedrock": {
        "default": {
            "BIG_MODEL": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "MODEL": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
            "SMALL_MODEL": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        }
    },
    "google-gemini": {
        "default": {
            "BIG_MODEL": "gemini-2.0-flash-exp",
            "MODEL": "gemini-2.0-flash-exp",
            "SMALL_MODEL": "gemini-1.5-flash",
        }
    },
    "google-anthropic": {
        "default": {
            "BIG_MODEL": "claude-3-7-sonnet@20250219",
            "MODEL": "claude-3-5-haiku@20241022",
            "SMALL_MODEL": "claude-3-5-haiku@20241022",
        }
    },
}