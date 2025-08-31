from reson.services.inference_clients import InferenceProvider

BIG_MODEL = "BIG_MODEL"
MODEL = "MODEL"
SMALL_MODEL = "SMALL_MODEL"

MODEL_CONFIGURATION = {
    InferenceProvider.OPENROUTER.value: {
        "default": {
            "BIG_MODEL": "anthropic/claude-3.7-sonnet",
            "MODEL": "anthropic/claude-3.5-haiku",
            "SMALL_MODEL": "anthropic/claude-3-haiku",
        },
        InferenceProvider.ANTHROPIC.value: {
            "BIG_MODEL": "anthropic/claude-3.7-sonnet",
            "MODEL": "anthropic/claude-3.5-haiku",
            "SMALL_MODEL": "anthropic/claude-3-haiku",
        },
        InferenceProvider.OPENAI.value: {
            "BIG_MODEL": "openai/o3",
            "MODEL": "openai/o4-mini",
            "SMALL_MODEL": "openai/o4-mini",
        },
    },
    InferenceProvider.ANTHROPIC.value: {
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
    InferenceProvider.BEDROCK.value: {
        "default": {
            "BIG_MODEL": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "MODEL": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
            "SMALL_MODEL": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        }
    },
    InferenceProvider.GOOGLE_GENAI.value: {
        "default": {
            "BIG_MODEL": "gemini-2.0-flash-exp",
            "MODEL": "gemini-2.0-flash-exp",
            "SMALL_MODEL": "gemini-1.5-flash",
        }
    },
    InferenceProvider.GOOGLE_ANTHROPIC.value: {
        "default": {
            "BIG_MODEL": "claude-3-7-sonnet@20250219",
            "MODEL": "claude-3-5-haiku@20241022",
            "SMALL_MODEL": "claude-3-5-haiku@20241022",
        }
    },
}
