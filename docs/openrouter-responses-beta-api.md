---
title: Responses API Beta
subtitle: OpenAI-compatible Responses API (Beta)
headline: OpenRouter Responses API Beta
canonical-url: 'https://openrouter.ai/docs/api/reference/responses/overview'
'og:site_name': OpenRouter Documentation
'og:title': OpenRouter Responses API Beta - OpenAI-Compatible Documentation
'og:description': >-
  Beta version of OpenRouter's OpenAI-compatible Responses API. Stateless
  transformation layer with support for reasoning, tool calling, and web search.
'og:image':
  type: url
  value: >-
    https://openrouter.ai/dynamic-og?title=Responses%20API%20Beta&description=OpenAI-compatible%20stateless%20API
'og:image:width': 1200
'og:image:height': 630
'twitter:card': summary_large_image
'twitter:site': '@OpenRouterAI'
noindex: false
nofollow: false
---

<Warning title="Beta API">
  This API is in **beta stage** and may have breaking changes. Use with caution in production environments.
</Warning>

<Info title="Stateless Only">
  This API is **stateless** - each request is independent and no conversation state is persisted between requests. You must include the full conversation history in each request.
</Info>

OpenRouter's Responses API Beta provides OpenAI-compatible access to multiple AI models through a unified interface, designed to be a drop-in replacement for OpenAI's Responses API. This stateless API offers enhanced capabilities including reasoning, tool calling, and web search integration, with each request being independent and no server-side state persisted.

## Base URL

```
https://openrouter.ai/api/v1/responses
```

## Authentication

All requests require authentication using your OpenRouter API key:

<CodeGroup>

```typescript title="TypeScript"
const response = await fetch('https://openrouter.ai/api/v1/responses', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'openai/o4-mini',
    input: 'Hello, world!',
  }),
});
```

```python title="Python"
import requests

response = requests.post(
    'https://openrouter.ai/api/v1/responses',
    headers={
        'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
        'Content-Type': 'application/json',
    },
    json={
        'model': 'openai/o4-mini',
        'input': 'Hello, world!',
    }
)
```

```bash title="cURL"
curl -X POST https://openrouter.ai/api/v1/responses \
  -H "Authorization: Bearer YOUR_OPENROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/o4-mini",
    "input": "Hello, world!"
  }'
```

</CodeGroup>

## Core Features

### [Basic Usage](./basic-usage)
Learn the fundamentals of making requests with simple text input and handling responses.

### [Reasoning](./reasoning)
Access advanced reasoning capabilities with configurable effort levels and encrypted reasoning chains.

### [Tool Calling](./tool-calling)
Integrate function calling with support for parallel execution and complex tool interactions.

### [Web Search](./web-search)
Enable web search capabilities with real-time information retrieval and citation annotations.


## Error Handling

The API returns structured error responses:

```json
{
  "error": {
    "code": "invalid_prompt",
    "message": "Missing required parameter: 'model'."
  },
  "metadata": null
}
```

For comprehensive error handling guidance, see [Error Handling](./error-handling).

## Rate Limits

Standard OpenRouter rate limits apply. See [API Limits](/docs/api-reference/limits) for details.
---
title: Basic Usage
subtitle: Getting started with the Responses API Beta
headline: Responses API Beta Basic Usage | Simple Text Requests
canonical-url: 'https://openrouter.ai/docs/api/reference/responses/basic-usage'
'og:site_name': OpenRouter Documentation
'og:title': Responses API Beta Basic Usage - Simple Text Requests
'og:description': >-
  Learn the basics of OpenRouter's Responses API Beta with simple text input
  examples and response handling.
'og:image':
  type: url
  value: >-
    https://openrouter.ai/dynamic-og?title=Responses%20API%20Basic%20Usage&description=Simple%20text%20requests%20and%20responses
'og:image:width': 1200
'og:image:height': 630
'twitter:card': summary_large_image
'twitter:site': '@OpenRouterAI'
noindex: false
nofollow: false
---

<Warning title="Beta API">
  This API is in **beta stage** and may have breaking changes.
</Warning>

The Responses API Beta supports both simple string input and structured message arrays, making it easy to get started with basic text generation.

## Simple String Input

The simplest way to use the API is with a string input:

<CodeGroup>

```typescript title="TypeScript"
const response = await fetch('https://openrouter.ai/api/v1/responses', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'openai/o4-mini',
    input: 'What is the meaning of life?',
    max_output_tokens: 9000,
  }),
});

const result = await response.json();
console.log(result);
```

```python title="Python"
import requests

response = requests.post(
    'https://openrouter.ai/api/v1/responses',
    headers={
        'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
        'Content-Type': 'application/json',
    },
    json={
        'model': 'openai/o4-mini',
        'input': 'What is the meaning of life?',
        'max_output_tokens': 9000,
    }
)

result = response.json()
print(result)
```

```bash title="cURL"
curl -X POST https://openrouter.ai/api/v1/responses \
  -H "Authorization: Bearer YOUR_OPENROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/o4-mini",
    "input": "What is the meaning of life?",
    "max_output_tokens": 9000
  }'
```

</CodeGroup>

## Structured Message Input

For more complex conversations, use the message array format:

<CodeGroup>

```typescript title="TypeScript"
const response = await fetch('https://openrouter.ai/api/v1/responses', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'openai/o4-mini',
    input: [
      {
        type: 'message',
        role: 'user',
        content: [
          {
            type: 'input_text',
            text: 'Tell me a joke about programming',
          },
        ],
      },
    ],
    max_output_tokens: 9000,
  }),
});

const result = await response.json();
```

```python title="Python"
import requests

response = requests.post(
    'https://openrouter.ai/api/v1/responses',
    headers={
        'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
        'Content-Type': 'application/json',
    },
    json={
        'model': 'openai/o4-mini',
        'input': [
            {
                'type': 'message',
                'role': 'user',
                'content': [
                    {
                        'type': 'input_text',
                        'text': 'Tell me a joke about programming',
                    },
                ],
            },
        ],
        'max_output_tokens': 9000,
    }
)

result = response.json()
```

```bash title="cURL"
curl -X POST https://openrouter.ai/api/v1/responses \
  -H "Authorization: Bearer YOUR_OPENROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/o4-mini",
    "input": [
      {
        "type": "message",
        "role": "user",
        "content": [
          {
            "type": "input_text",
            "text": "Tell me a joke about programming"
          }
        ]
      }
    ],
    "max_output_tokens": 9000
  }'
```

</CodeGroup>

## Response Format

The API returns a structured response with the generated content:

```json
{
  "id": "resp_1234567890",
  "object": "response",
  "created_at": 1234567890,
  "model": "openai/o4-mini",
  "output": [
    {
      "type": "message",
      "id": "msg_abc123",
      "status": "completed",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "The meaning of life is a philosophical question that has been pondered for centuries...",
          "annotations": []
        }
      ]
    }
  ],
  "usage": {
    "input_tokens": 12,
    "output_tokens": 45,
    "total_tokens": 57
  },
  "status": "completed"
}
```

## Streaming Responses

Enable streaming for real-time response generation:

<CodeGroup>

```typescript title="TypeScript"
const response = await fetch('https://openrouter.ai/api/v1/responses', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'openai/o4-mini',
    input: 'Write a short story about AI',
    stream: true,
    max_output_tokens: 9000,
  }),
});

const reader = response.body?.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');

  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = line.slice(6);
      if (data === '[DONE]') return;

      try {
        const parsed = JSON.parse(data);
        console.log(parsed);
      } catch (e) {
        // Skip invalid JSON
      }
    }
  }
}
```

```python title="Python"
import requests
import json

response = requests.post(
    'https://openrouter.ai/api/v1/responses',
    headers={
        'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
        'Content-Type': 'application/json',
    },
    json={
        'model': 'openai/o4-mini',
        'input': 'Write a short story about AI',
        'stream': True,
        'max_output_tokens': 9000,
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        line_str = line.decode('utf-8')
        if line_str.startswith('data: '):
            data = line_str[6:]
            if data == '[DONE]':
                break
            try:
                parsed = json.loads(data)
                print(parsed)
            except json.JSONDecodeError:
                continue
```

</CodeGroup>

### Example Streaming Output

The streaming response returns Server-Sent Events (SSE) chunks:

```
data: {"type":"response.created","response":{"id":"resp_1234567890","object":"response","status":"in_progress"}}

data: {"type":"response.output_item.added","response_id":"resp_1234567890","output_index":0,"item":{"type":"message","id":"msg_abc123","role":"assistant","status":"in_progress","content":[]}}

data: {"type":"response.content_part.added","response_id":"resp_1234567890","output_index":0,"content_index":0,"part":{"type":"output_text","text":""}}

data: {"type":"response.content_part.delta","response_id":"resp_1234567890","output_index":0,"content_index":0,"delta":"Once"}

data: {"type":"response.content_part.delta","response_id":"resp_1234567890","output_index":0,"content_index":0,"delta":" upon"}

data: {"type":"response.content_part.delta","response_id":"resp_1234567890","output_index":0,"content_index":0,"delta":" a"}

data: {"type":"response.content_part.delta","response_id":"resp_1234567890","output_index":0,"content_index":0,"delta":" time"}

data: {"type":"response.output_item.done","response_id":"resp_1234567890","output_index":0,"item":{"type":"message","id":"msg_abc123","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Once upon a time, in a world where artificial intelligence had become as common as smartphones..."}]}}

data: {"type":"response.done","response":{"id":"resp_1234567890","object":"response","status":"completed","usage":{"input_tokens":12,"output_tokens":45,"total_tokens":57}}}

data: [DONE]
```

## Common Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | **Required.** Model to use (e.g., `openai/o4-mini`) |
| `input` | string or array | **Required.** Text or message array |
| `stream` | boolean | Enable streaming responses (default: false) |
| `max_output_tokens` | integer | Maximum tokens to generate |
| `temperature` | number | Sampling temperature (0-2) |
| `top_p` | number | Nucleus sampling parameter (0-1) |

## Error Handling

Handle common errors gracefully:

<CodeGroup>

```typescript title="TypeScript"
try {
  const response = await fetch('https://openrouter.ai/api/v1/responses', {
    method: 'POST',
    headers: {
      'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'openai/o4-mini',
      input: 'Hello, world!',
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    console.error('API Error:', error.error.message);
    return;
  }

  const result = await response.json();
  console.log(result);
} catch (error) {
  console.error('Network Error:', error);
}
```

```python title="Python"
import requests

try:
    response = requests.post(
        'https://openrouter.ai/api/v1/responses',
        headers={
            'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
            'Content-Type': 'application/json',
        },
        json={
            'model': 'openai/o4-mini',
            'input': 'Hello, world!',
        }
    )

    if response.status_code != 200:
        error = response.json()
        print(f"API Error: {error['error']['message']}")
    else:
        result = response.json()
        print(result)

except requests.RequestException as e:
    print(f"Network Error: {e}")
```

</CodeGroup>

## Multiple Turn Conversations

Since the Responses API Beta is stateless, you must include the full conversation history in each request to maintain context:

<CodeGroup>

```typescript title="TypeScript"
// First request
const firstResponse = await fetch('https://openrouter.ai/api/beta/responses', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'openai/o4-mini',
    input: [
      {
        type: 'message',
        role: 'user',
        content: [
          {
            type: 'input_text',
            text: 'What is the capital of France?',
          },
        ],
      },
    ],
    max_output_tokens: 9000,
  }),
});

const firstResult = await firstResponse.json();

// Second request - include previous conversation
const secondResponse = await fetch('https://openrouter.ai/api/beta/responses', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'openai/o4-mini',
    input: [
      {
        type: 'message',
        role: 'user',
        content: [
          {
            type: 'input_text',
            text: 'What is the capital of France?',
          },
        ],
      },
      {
        type: 'message',
        role: 'assistant',
        id: 'msg_abc123',
        status: 'completed',
        content: [
          {
            type: 'output_text',
            text: 'The capital of France is Paris.',
            annotations: []
          }
        ]
      },
      {
        type: 'message',
        role: 'user',
        content: [
          {
            type: 'input_text',
            text: 'What is the population of that city?',
          },
        ],
      },
    ],
    max_output_tokens: 9000,
  }),
});

const secondResult = await secondResponse.json();
```

```python title="Python"
import requests

# First request
first_response = requests.post(
    'https://openrouter.ai/api/v1/responses',
    headers={
        'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
        'Content-Type': 'application/json',
    },
    json={
        'model': 'openai/o4-mini',
        'input': [
            {
                'type': 'message',
                'role': 'user',
                'content': [
                    {
                        'type': 'input_text',
                        'text': 'What is the capital of France?',
                    },
                ],
            },
        ],
        'max_output_tokens': 9000,
    }
)

first_result = first_response.json()

# Second request - include previous conversation
second_response = requests.post(
    'https://openrouter.ai/api/v1/responses',
    headers={
        'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
        'Content-Type': 'application/json',
    },
    json={
        'model': 'openai/o4-mini',
        'input': [
            {
                'type': 'message',
                'role': 'user',
                'content': [
                    {
                        'type': 'input_text',
                        'text': 'What is the capital of France?',
                    },
                ],
            },
            {
                'type': 'message',
                'role': 'assistant',
                'id': 'msg_abc123',
                'status': 'completed',
                'content': [
                    {
                        'type': 'output_text',
                        'text': 'The capital of France is Paris.',
                        'annotations': []
                    }
                ]
            },
            {
                'type': 'message',
                'role': 'user',
                'content': [
                    {
                        'type': 'input_text',
                        'text': 'What is the population of that city?',
                    },
                ],
            },
        ],
        'max_output_tokens': 9000,
    }
)

second_result = second_response.json()
```

</CodeGroup>

<Info title="Required Fields">
  The `id` and `status` fields are required for any `assistant` role messages included in the conversation history.
</Info>

<Info title="Conversation History">
  Always include the complete conversation history in each request. The API does not store previous messages, so context must be maintained client-side.
</Info>

## Next Steps

- Learn about [Reasoning](./reasoning) capabilities
- Explore [Tool Calling](./tool-calling) functionality
- Try [Web Search](./web-search) integration
---
title: Reasoning
subtitle: Advanced reasoning capabilities with the Responses API Beta
headline: Responses API Beta Reasoning | Advanced AI Reasoning Capabilities
canonical-url: 'https://openrouter.ai/docs/api/reference/responses/reasoning'
'og:site_name': OpenRouter Documentation
'og:title': Responses API Beta Reasoning - Advanced AI Reasoning
'og:description': >-
  Access advanced reasoning capabilities with configurable effort levels and
  encrypted reasoning chains using OpenRouter's Responses API Beta.
'og:image':
  type: url
  value: >-
    https://openrouter.ai/dynamic-og?title=Responses%20API%20Reasoning&description=Advanced%20AI%20reasoning%20capabilities
'og:image:width': 1200
'og:image:height': 630
'twitter:card': summary_large_image
'twitter:site': '@OpenRouterAI'
noindex: false
nofollow: false
---

<Warning title="Beta API">
  This API is in **beta stage** and may have breaking changes.
</Warning>

The Responses API Beta supports advanced reasoning capabilities, allowing models to show their internal reasoning process with configurable effort levels.

## Reasoning Configuration

Configure reasoning behavior using the `reasoning` parameter:

<CodeGroup>

```typescript title="TypeScript"
const response = await fetch('https://openrouter.ai/api/v1/responses', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'openai/o4-mini',
    input: 'What is the meaning of life?',
    reasoning: {
      effort: 'high'
    },
    max_output_tokens: 9000,
  }),
});

const result = await response.json();
console.log(result);
```

```python title="Python"
import requests

response = requests.post(
    'https://openrouter.ai/api/v1/responses',
    headers={
        'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
        'Content-Type': 'application/json',
    },
    json={
        'model': 'openai/o4-mini',
        'input': 'What is the meaning of life?',
        'reasoning': {
            'effort': 'high'
        },
        'max_output_tokens': 9000,
    }
)

result = response.json()
print(result)
```

```bash title="cURL"
curl -X POST https://openrouter.ai/api/v1/responses \
  -H "Authorization: Bearer YOUR_OPENROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/o4-mini",
    "input": "What is the meaning of life?",
    "reasoning": {
      "effort": "high"
    },
    "max_output_tokens": 9000
  }'
```

</CodeGroup>

## Reasoning Effort Levels

The `effort` parameter controls how much computational effort the model puts into reasoning:

| Effort Level | Description |
|--------------|-------------|
| `minimal` | Basic reasoning with minimal computational effort |
| `low` | Light reasoning for simple problems |
| `medium` | Balanced reasoning for moderate complexity |
| `high` | Deep reasoning for complex problems |

## Complex Reasoning Example

For complex mathematical or logical problems:

<CodeGroup>

```typescript title="TypeScript"
const response = await fetch('https://openrouter.ai/api/v1/responses', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'openai/o4-mini',
    input: [
      {
        type: 'message',
        role: 'user',
        content: [
          {
            type: 'input_text',
            text: 'Was 1995 30 years ago? Please show your reasoning.',
          },
        ],
      },
    ],
    reasoning: {
      effort: 'high'
    },
    max_output_tokens: 9000,
  }),
});

const result = await response.json();
console.log(result);
```

```python title="Python"
import requests

response = requests.post(
    'https://openrouter.ai/api/v1/responses',
    headers={
        'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
        'Content-Type': 'application/json',
    },
    json={
        'model': 'openai/o4-mini',
        'input': [
            {
                'type': 'message',
                'role': 'user',
                'content': [
                    {
                        'type': 'input_text',
                        'text': 'Was 1995 30 years ago? Please show your reasoning.',
                    },
                ],
            },
        ],
        'reasoning': {
            'effort': 'high'
        },
        'max_output_tokens': 9000,
    }
)

result = response.json()
print(result)
```

</CodeGroup>

## Reasoning in Conversation Context

Include reasoning in multi-turn conversations:

<CodeGroup>

```typescript title="TypeScript"
const response = await fetch('https://openrouter.ai/api/v1/responses', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'openai/o4-mini',
    input: [
      {
        type: 'message',
        role: 'user',
        content: [
          {
            type: 'input_text',
            text: 'What is your favorite color?',
          },
        ],
      },
      {
        type: 'message',
        role: 'assistant',
        id: 'msg_abc123',
        status: 'completed',
        content: [
          {
            type: 'output_text',
            text: "I don't have a favorite color.",
            annotations: []
          }
        ]
      },
      {
        type: 'message',
        role: 'user',
        content: [
          {
            type: 'input_text',
            text: 'How many Earths can fit on Mars?',
          },
        ],
      },
    ],
    reasoning: {
      effort: 'high'
    },
    max_output_tokens: 9000,
  }),
});

const result = await response.json();
console.log(result);
```

```python title="Python"
import requests

response = requests.post(
    'https://openrouter.ai/api/v1/responses',
    headers={
        'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
        'Content-Type': 'application/json',
    },
    json={
        'model': 'openai/o4-mini',
        'input': [
            {
                'type': 'message',
                'role': 'user',
                'content': [
                    {
                        'type': 'input_text',
                        'text': 'What is your favorite color?',
                    },
                ],
            },
            {
                'type': 'message',
                'role': 'assistant',
                'id': 'msg_abc123',
                'status': 'completed',
                'content': [
                    {
                        'type': 'output_text',
                        'text': "I don't have a favorite color.",
                        'annotations': []
                    }
                ]
            },
            {
                'type': 'message',
                'role': 'user',
                'content': [
                    {
                        'type': 'input_text',
                        'text': 'How many Earths can fit on Mars?',
                    },
                ],
            },
        ],
        'reasoning': {
            'effort': 'high'
        },
        'max_output_tokens': 9000,
    }
)

result = response.json()
print(result)
```

</CodeGroup>

## Streaming Reasoning

Enable streaming to see reasoning develop in real-time:

<CodeGroup>

```typescript title="TypeScript"
const response = await fetch('https://openrouter.ai/api/v1/responses', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'openai/o4-mini',
    input: 'Solve this step by step: If a train travels 60 mph for 2.5 hours, how far does it go?',
    reasoning: {
      effort: 'medium'
    },
    stream: true,
    max_output_tokens: 9000,
  }),
});

const reader = response.body?.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');

  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = line.slice(6);
      if (data === '[DONE]') return;

      try {
        const parsed = JSON.parse(data);
        if (parsed.type === 'response.reasoning.delta') {
          console.log('Reasoning:', parsed.delta);
        }
      } catch (e) {
        // Skip invalid JSON
      }
    }
  }
}
```

```python title="Python"
import requests
import json

response = requests.post(
    'https://openrouter.ai/api/v1/responses',
    headers={
        'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
        'Content-Type': 'application/json',
    },
    json={
        'model': 'openai/o4-mini',
        'input': 'Solve this step by step: If a train travels 60 mph for 2.5 hours, how far does it go?',
        'reasoning': {
            'effort': 'medium'
        },
        'stream': True,
        'max_output_tokens': 9000,
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        line_str = line.decode('utf-8')
        if line_str.startswith('data: '):
            data = line_str[6:]
            if data == '[DONE]':
                break
            try:
                parsed = json.loads(data)
                if parsed.get('type') == 'response.reasoning.delta':
                    print(f"Reasoning: {parsed.get('delta', '')}")
            except json.JSONDecodeError:
                continue
```

</CodeGroup>

## Response with Reasoning

When reasoning is enabled, the response includes reasoning information:

```json
{
  "id": "resp_1234567890",
  "object": "response",
  "created_at": 1234567890,
  "model": "openai/o4-mini",
  "output": [
    {
      "type": "reasoning",
      "id": "rs_abc123",
      "encrypted_content": "gAAAAABotI9-FK1PbhZhaZk4yMrZw3XDI1AWFaKb9T0NQq7LndK6zaRB...",
      "summary": [
        "First, I need to determine the current year",
        "Then calculate the difference from 1995",
        "Finally, compare that to 30 years"
      ]
    },
    {
      "type": "message",
      "id": "msg_xyz789",
      "status": "completed",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "Yes. In 2025, 1995 was 30 years ago. In fact, as of today (Aug 31, 2025), it's exactly 30 years since Aug 31, 1995.",
          "annotations": []
        }
      ]
    }
  ],
  "usage": {
    "input_tokens": 15,
    "output_tokens": 85,
    "output_tokens_details": {
      "reasoning_tokens": 45
    },
    "total_tokens": 100
  },
  "status": "completed"
}
```

## Best Practices

1. **Choose appropriate effort levels**: Use `high` for complex problems, `low` for simple tasks
2. **Consider token usage**: Reasoning increases token consumption
3. **Use streaming**: For long reasoning chains, streaming provides better user experience
4. **Include context**: Provide sufficient context for the model to reason effectively

## Next Steps

- Explore [Tool Calling](./tool-calling) with reasoning
- Learn about [Web Search](./web-search) integration
- Review [Basic Usage](./basic-usage) fundamentals
---
title: Tool Calling
subtitle: Function calling and tool integration with the Responses API Beta
headline: Responses API Beta Tool Calling | Function Calling Integration
canonical-url: 'https://openrouter.ai/docs/api/reference/responses/tool-calling'
'og:site_name': OpenRouter Documentation
'og:title': Responses API Beta Tool Calling - Function Calling Integration
'og:description': >-
  Integrate function calling with support for parallel execution and complex
  tool interactions using OpenRouter's Responses API Beta.
'og:image':
  type: url
  value: >-
    https://openrouter.ai/dynamic-og?title=Responses%20API%20Tool%20Calling&description=Function%20calling%20integration
'og:image:width': 1200
'og:image:height': 630
'twitter:card': summary_large_image
'twitter:site': '@OpenRouterAI'
noindex: false
nofollow: false
---

<Warning title="Beta API">
  This API is in **beta stage** and may have breaking changes.
</Warning>

The Responses API Beta supports comprehensive tool calling capabilities, allowing models to call functions, execute tools in parallel, and handle complex multi-step workflows.

## Basic Tool Definition

Define tools using the OpenAI function calling format:

<CodeGroup>

```typescript title="TypeScript"
const weatherTool = {
  type: 'function' as const,
  name: 'get_weather',
  description: 'Get the current weather in a location',
  strict: null,
  parameters: {
    type: 'object',
    properties: {
      location: {
        type: 'string',
        description: 'The city and state, e.g. San Francisco, CA',
      },
      unit: {
        type: 'string',
        enum: ['celsius', 'fahrenheit'],
      },
    },
    required: ['location'],
  },
};

const response = await fetch('https://openrouter.ai/api/v1/responses', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'openai/o4-mini',
    input: [
      {
        type: 'message',
        role: 'user',
        content: [
          {
            type: 'input_text',
            text: 'What is the weather in San Francisco?',
          },
        ],
      },
    ],
    tools: [weatherTool],
    tool_choice: 'auto',
    max_output_tokens: 9000,
  }),
});

const result = await response.json();
console.log(result);
```

```python title="Python"
import requests

weather_tool = {
    'type': 'function',
    'name': 'get_weather',
    'description': 'Get the current weather in a location',
    'strict': None,
    'parameters': {
        'type': 'object',
        'properties': {
            'location': {
                'type': 'string',
                'description': 'The city and state, e.g. San Francisco, CA',
            },
            'unit': {
                'type': 'string',
                'enum': ['celsius', 'fahrenheit'],
            },
        },
        'required': ['location'],
    },
}

response = requests.post(
    'https://openrouter.ai/api/v1/responses',
    headers={
        'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
        'Content-Type': 'application/json',
    },
    json={
        'model': 'openai/o4-mini',
        'input': [
            {
                'type': 'message',
                'role': 'user',
                'content': [
                    {
                        'type': 'input_text',
                        'text': 'What is the weather in San Francisco?',
                    },
                ],
            },
        ],
        'tools': [weather_tool],
        'tool_choice': 'auto',
        'max_output_tokens': 9000,
    }
)

result = response.json()
print(result)
```

```bash title="cURL"
curl -X POST https://openrouter.ai/api/v1/responses \
  -H "Authorization: Bearer YOUR_OPENROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/o4-mini",
    "input": [
      {
        "type": "message",
        "role": "user",
        "content": [
          {
            "type": "input_text",
            "text": "What is the weather in San Francisco?"
          }
        ]
      }
    ],
    "tools": [
      {
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather in a location",
        "strict": null,
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state, e.g. San Francisco, CA"
            },
            "unit": {
              "type": "string",
              "enum": ["celsius", "fahrenheit"]
            }
          },
          "required": ["location"]
        }
      }
    ],
    "tool_choice": "auto",
    "max_output_tokens": 9000
  }'
```

</CodeGroup>

## Tool Choice Options

Control when and how tools are called:

| Tool Choice | Description |
|-------------|-------------|
| `auto` | Model decides whether to call tools |
| `none` | Model will not call any tools |
| `{type: 'function', name: 'tool_name'}` | Force specific tool call |

### Force Specific Tool

<CodeGroup>

```typescript title="TypeScript"
const response = await fetch('https://openrouter.ai/api/v1/responses', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'openai/o4-mini',
    input: [
      {
        type: 'message',
        role: 'user',
        content: [
          {
            type: 'input_text',
            text: 'Hello, how are you?',
          },
        ],
      },
    ],
    tools: [weatherTool],
    tool_choice: { type: 'function', name: 'get_weather' },
    max_output_tokens: 9000,
  }),
});
```

```python title="Python"
import requests

response = requests.post(
    'https://openrouter.ai/api/v1/responses',
    headers={
        'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
        'Content-Type': 'application/json',
    },
    json={
        'model': 'openai/o4-mini',
        'input': [
            {
                'type': 'message',
                'role': 'user',
                'content': [
                    {
                        'type': 'input_text',
                        'text': 'Hello, how are you?',
                    },
                ],
            },
        ],
        'tools': [weather_tool],
        'tool_choice': {'type': 'function', 'name': 'get_weather'},
        'max_output_tokens': 9000,
    }
)
```

</CodeGroup>

### Disable Tool Calling

<CodeGroup>

```typescript title="TypeScript"
const response = await fetch('https://openrouter.ai/api/v1/responses', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'openai/o4-mini',
    input: [
      {
        type: 'message',
        role: 'user',
        content: [
          {
            type: 'input_text',
            text: 'What is the weather in Paris?',
          },
        ],
      },
    ],
    tools: [weatherTool],
    tool_choice: 'none',
    max_output_tokens: 9000,
  }),
});
```

```python title="Python"
import requests

response = requests.post(
    'https://openrouter.ai/api/v1/responses',
    headers={
        'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
        'Content-Type': 'application/json',
    },
    json={
        'model': 'openai/o4-mini',
        'input': [
            {
                'type': 'message',
                'role': 'user',
                'content': [
                    {
                        'type': 'input_text',
                        'text': 'What is the weather in Paris?',
                    },
                ],
            },
        ],
        'tools': [weather_tool],
        'tool_choice': 'none',
        'max_output_tokens': 9000,
    }
)
```

</CodeGroup>

## Multiple Tools

Define multiple tools for complex workflows:

<CodeGroup>

```typescript title="TypeScript"
const calculatorTool = {
  type: 'function' as const,
  name: 'calculate',
  description: 'Perform mathematical calculations',
  strict: null,
  parameters: {
    type: 'object',
    properties: {
      expression: {
        type: 'string',
        description: 'The mathematical expression to evaluate',
      },
    },
    required: ['expression'],
  },
};

const response = await fetch('https://openrouter.ai/api/v1/responses', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'openai/o4-mini',
    input: [
      {
        type: 'message',
        role: 'user',
        content: [
          {
            type: 'input_text',
            text: 'What is 25 * 4?',
          },
        ],
      },
    ],
    tools: [weatherTool, calculatorTool],
    tool_choice: 'auto',
    max_output_tokens: 9000,
  }),
});
```

```python title="Python"
calculator_tool = {
    'type': 'function',
    'name': 'calculate',
    'description': 'Perform mathematical calculations',
    'strict': None,
    'parameters': {
        'type': 'object',
        'properties': {
            'expression': {
                'type': 'string',
                'description': 'The mathematical expression to evaluate',
            },
        },
        'required': ['expression'],
    },
}

response = requests.post(
    'https://openrouter.ai/api/v1/responses',
    headers={
        'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
        'Content-Type': 'application/json',
    },
    json={
        'model': 'openai/o4-mini',
        'input': [
            {
                'type': 'message',
                'role': 'user',
                'content': [
                    {
                        'type': 'input_text',
                        'text': 'What is 25 * 4?',
                    },
                ],
            },
        ],
        'tools': [weather_tool, calculator_tool],
        'tool_choice': 'auto',
        'max_output_tokens': 9000,
    }
)
```

</CodeGroup>

## Parallel Tool Calls

The API supports parallel execution of multiple tools:

<CodeGroup>

```typescript title="TypeScript"
const response = await fetch('https://openrouter.ai/api/v1/responses', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'openai/o4-mini',
    input: [
      {
        type: 'message',
        role: 'user',
        content: [
          {
            type: 'input_text',
            text: 'Calculate 10*5 and also tell me the weather in Miami',
          },
        ],
      },
    ],
    tools: [weatherTool, calculatorTool],
    tool_choice: 'auto',
    max_output_tokens: 9000,
  }),
});

const result = await response.json();
console.log(result);
```

```python title="Python"
import requests

response = requests.post(
    'https://openrouter.ai/api/v1/responses',
    headers={
        'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
        'Content-Type': 'application/json',
    },
    json={
        'model': 'openai/o4-mini',
        'input': [
            {
                'type': 'message',
                'role': 'user',
                'content': [
                    {
                        'type': 'input_text',
                        'text': 'Calculate 10*5 and also tell me the weather in Miami',
                    },
                ],
            },
        ],
        'tools': [weather_tool, calculator_tool],
        'tool_choice': 'auto',
        'max_output_tokens': 9000,
    }
)

result = response.json()
print(result)
```

</CodeGroup>

## Tool Call Response

When tools are called, the response includes function call information:

```json
{
  "id": "resp_1234567890",
  "object": "response",
  "created_at": 1234567890,
  "model": "openai/o4-mini",
  "output": [
    {
      "type": "function_call",
      "id": "fc_abc123",
      "call_id": "call_xyz789",
      "name": "get_weather",
      "arguments": "{\"location\":\"San Francisco, CA\"}"
    }
  ],
  "usage": {
    "input_tokens": 45,
    "output_tokens": 25,
    "total_tokens": 70
  },
  "status": "completed"
}
```

## Tool Responses in Conversation

Include tool responses in follow-up requests:

<CodeGroup>

```typescript title="TypeScript"
const response = await fetch('https://openrouter.ai/api/v1/responses', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'openai/o4-mini',
    input: [
      {
        type: 'message',
        role: 'user',
        content: [
          {
            type: 'input_text',
            text: 'What is the weather in Boston?',
          },
        ],
      },
      {
        type: 'function_call',
        id: 'fc_1',
        call_id: 'call_123',
        name: 'get_weather',
        arguments: JSON.stringify({ location: 'Boston, MA' }),
      },
      {
        type: 'function_call_output',
        id: 'fc_output_1',
        call_id: 'call_123',
        output: JSON.stringify({ temperature: '72°F', condition: 'Sunny' }),
      },
      {
        type: 'message',
        role: 'assistant',
        id: 'msg_abc123',
        status: 'completed',
        content: [
          {
            type: 'output_text',
            text: 'The weather in Boston is currently 72°F and sunny. This looks like perfect weather for a picnic!',
            annotations: []
          }
        ]
      },
      {
        type: 'message',
        role: 'user',
        content: [
          {
            type: 'input_text',
            text: 'Is that good weather for a picnic?',
          },
        ],
      },
    ],
    max_output_tokens: 9000,
  }),
});
```

```python title="Python"
import requests

response = requests.post(
    'https://openrouter.ai/api/v1/responses',
    headers={
        'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
        'Content-Type': 'application/json',
    },
    json={
        'model': 'openai/o4-mini',
        'input': [
            {
                'type': 'message',
                'role': 'user',
                'content': [
                    {
                        'type': 'input_text',
                        'text': 'What is the weather in Boston?',
                    },
                ],
            },
            {
                'type': 'function_call',
                'id': 'fc_1',
                'call_id': 'call_123',
                'name': 'get_weather',
                'arguments': '{"location": "Boston, MA"}',
            },
            {
                'type': 'function_call_output',
                'id': 'fc_output_1',
                'call_id': 'call_123',
                'output': '{"temperature": "72°F", "condition": "Sunny"}',
            },
            {
                'type': 'message',
                'role': 'assistant',
                'id': 'msg_abc123',
                'status': 'completed',
                'content': [
                    {
                        'type': 'output_text',
                        'text': 'The weather in Boston is currently 72°F and sunny. This looks like perfect weather for a picnic!',
                        'annotations': []
                    }
                ]
            },
            {
                'type': 'message',
                'role': 'user',
                'content': [
                    {
                        'type': 'input_text',
                        'text': 'Is that good weather for a picnic?',
                    },
                ],
            },
        ],
        'max_output_tokens': 9000,
    }
)
```

</CodeGroup>

<Info title="Required Field">
  The `id` field is required for `function_call_output` objects when including tool responses in conversation history.
</Info>

## Streaming Tool Calls

Monitor tool calls in real-time with streaming:

<CodeGroup>

```typescript title="TypeScript"
const response = await fetch('https://openrouter.ai/api/v1/responses', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'openai/o4-mini',
    input: [
      {
        type: 'message',
        role: 'user',
        content: [
          {
            type: 'input_text',
            text: 'What is the weather like in Tokyo, Japan? Please check the weather.',
          },
        ],
      },
    ],
    tools: [weatherTool],
    tool_choice: 'auto',
    stream: true,
    max_output_tokens: 9000,
  }),
});

const reader = response.body?.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');

  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = line.slice(6);
      if (data === '[DONE]') return;

      try {
        const parsed = JSON.parse(data);
        if (parsed.type === 'response.output_item.added' &&
            parsed.item?.type === 'function_call') {
          console.log('Function call:', parsed.item.name);
        }
        if (parsed.type === 'response.function_call_arguments.done') {
          console.log('Arguments:', parsed.arguments);
        }
      } catch (e) {
        // Skip invalid JSON
      }
    }
  }
}
```

```python title="Python"
import requests
import json

response = requests.post(
    'https://openrouter.ai/api/v1/responses',
    headers={
        'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
        'Content-Type': 'application/json',
    },
    json={
        'model': 'openai/o4-mini',
        'input': [
            {
                'type': 'message',
                'role': 'user',
                'content': [
                    {
                        'type': 'input_text',
                        'text': 'What is the weather like in Tokyo, Japan? Please check the weather.',
                    },
                ],
            },
        ],
        'tools': [weather_tool],
        'tool_choice': 'auto',
        'stream': True,
        'max_output_tokens': 9000,
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        line_str = line.decode('utf-8')
        if line_str.startswith('data: '):
            data = line_str[6:]
            if data == '[DONE]':
                break
            try:
                parsed = json.loads(data)
                if (parsed.get('type') == 'response.output_item.added' and
                    parsed.get('item', {}).get('type') == 'function_call'):
                    print(f"Function call: {parsed['item']['name']}")
                if parsed.get('type') == 'response.function_call_arguments.done':
                    print(f"Arguments: {parsed.get('arguments', '')}")
            except json.JSONDecodeError:
                continue
```

</CodeGroup>

## Tool Validation

Ensure tool calls have proper structure:

```json
{
  "type": "function_call",
  "id": "fc_abc123",
  "call_id": "call_xyz789",
  "name": "get_weather",
  "arguments": "{\"location\":\"Seattle, WA\"}"
}
```

Required fields:
- `type`: Always "function_call"
- `id`: Unique identifier for the function call object
- `name`: Function name matching tool definition
- `arguments`: Valid JSON string with function parameters
- `call_id`: Unique identifier for the call

## Best Practices

1. **Clear descriptions**: Provide detailed function descriptions and parameter explanations
2. **Proper schemas**: Use valid JSON Schema for parameters
3. **Error handling**: Handle cases where tools might not be called
4. **Parallel execution**: Design tools to work independently when possible
5. **Conversation flow**: Include tool responses in follow-up requests for context

## Next Steps

- Learn about [Web Search](./web-search) integration
- Explore [Reasoning](./reasoning) with tools
- Review [Basic Usage](./basic-usage) fundamentals
---
title: Web Search
subtitle: Real-time web search integration with the Responses API Beta
headline: Responses API Beta Web Search | Real-time Information Retrieval
canonical-url: 'https://openrouter.ai/docs/api/reference/responses/web-search'
'og:site_name': OpenRouter Documentation
'og:title': Responses API Beta Web Search - Real-time Information Retrieval
'og:description': >-
  Enable web search capabilities with real-time information retrieval and
  citation annotations using OpenRouter's Responses API Beta.
'og:image':
  type: url
  value: >-
    https://openrouter.ai/dynamic-og?title=Responses%20API%20Web%20Search&description=Real-time%20information%20retrieval
'og:image:width': 1200
'og:image:height': 630
'twitter:card': summary_large_image
'twitter:site': '@OpenRouterAI'
noindex: false
nofollow: false
---

<Warning title="Beta API">
  This API is in **beta stage** and may have breaking changes.
</Warning>

The Responses API Beta supports web search integration, allowing models to access real-time information from the internet and provide responses with proper citations and annotations.

## Web Search Plugin

Enable web search using the `plugins` parameter:

<CodeGroup>

```typescript title="TypeScript"
const response = await fetch('https://openrouter.ai/api/v1/responses', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'openai/o4-mini',
    input: 'What is OpenRouter?',
    plugins: [{ id: 'web', max_results: 3 }],
    max_output_tokens: 9000,
  }),
});

const result = await response.json();
console.log(result);
```

```python title="Python"
import requests

response = requests.post(
    'https://openrouter.ai/api/v1/responses',
    headers={
        'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
        'Content-Type': 'application/json',
    },
    json={
        'model': 'openai/o4-mini',
        'input': 'What is OpenRouter?',
        'plugins': [{'id': 'web', 'max_results': 3}],
        'max_output_tokens': 9000,
    }
)

result = response.json()
print(result)
```

```bash title="cURL"
curl -X POST https://openrouter.ai/api/v1/responses \
  -H "Authorization: Bearer YOUR_OPENROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/o4-mini",
    "input": "What is OpenRouter?",
    "plugins": [{"id": "web", "max_results": 3}],
    "max_output_tokens": 9000
  }'
```

</CodeGroup>

## Plugin Configuration

Configure web search behavior:

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | string | **Required.** Must be "web" |
| `max_results` | integer | Maximum search results to retrieve (1-10) |

## Structured Message with Web Search

Use structured messages for more complex queries:

<CodeGroup>

```typescript title="TypeScript"
const response = await fetch('https://openrouter.ai/api/v1/responses', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'openai/o4-mini',
    input: [
      {
        type: 'message',
        role: 'user',
        content: [
          {
            type: 'input_text',
            text: 'What was a positive news story from today?',
          },
        ],
      },
    ],
    plugins: [{ id: 'web', max_results: 2 }],
    max_output_tokens: 9000,
  }),
});

const result = await response.json();
console.log(result);
```

```python title="Python"
import requests

response = requests.post(
    'https://openrouter.ai/api/v1/responses',
    headers={
        'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
        'Content-Type': 'application/json',
    },
    json={
        'model': 'openai/o4-mini',
        'input': [
            {
                'type': 'message',
                'role': 'user',
                'content': [
                    {
                        'type': 'input_text',
                        'text': 'What was a positive news story from today?',
                    },
                ],
            },
        ],
        'plugins': [{'id': 'web', 'max_results': 2}],
        'max_output_tokens': 9000,
    }
)

result = response.json()
print(result)
```

</CodeGroup>

## Online Model Variants

Some models have built-in web search capabilities using the `:online` variant:

<CodeGroup>

```typescript title="TypeScript"
const response = await fetch('https://openrouter.ai/api/v1/responses', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'openai/o4-mini:online',
    input: 'What was a positive news story from today?',
    max_output_tokens: 9000,
  }),
});

const result = await response.json();
console.log(result);
```

```python title="Python"
import requests

response = requests.post(
    'https://openrouter.ai/api/v1/responses',
    headers={
        'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
        'Content-Type': 'application/json',
    },
    json={
        'model': 'openai/o4-mini:online',
        'input': 'What was a positive news story from today?',
        'max_output_tokens': 9000,
    }
)

result = response.json()
print(result)
```

</CodeGroup>

## Response with Annotations

Web search responses include citation annotations:

```json
{
  "id": "resp_1234567890",
  "object": "response",
  "created_at": 1234567890,
  "model": "openai/o4-mini",
  "output": [
    {
      "type": "message",
      "id": "msg_abc123",
      "status": "completed",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "OpenRouter is a unified API for accessing multiple Large Language Model providers through a single interface. It allows developers to access 100+ AI models from providers like OpenAI, Anthropic, Google, and others with intelligent routing and automatic failover.",
          "annotations": [
            {
              "type": "url_citation",
              "url": "https://openrouter.ai/docs",
              "start_index": 0,
              "end_index": 85
            },
            {
              "type": "url_citation",
              "url": "https://openrouter.ai/models",
              "start_index": 120,
              "end_index": 180
            }
          ]
        }
      ]
    }
  ],
  "usage": {
    "input_tokens": 15,
    "output_tokens": 95,
    "total_tokens": 110
  },
  "status": "completed"
}
```

## Annotation Types

Web search responses can include different annotation types:

### URL Citation
```json
{
  "type": "url_citation",
  "url": "https://example.com/article",
  "start_index": 0,
  "end_index": 50
}
```


## Complex Search Queries

Handle multi-part search queries:

<CodeGroup>

```typescript title="TypeScript"
const response = await fetch('https://openrouter.ai/api/v1/responses', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'openai/o4-mini',
    input: [
      {
        type: 'message',
        role: 'user',
        content: [
          {
            type: 'input_text',
            text: 'Compare OpenAI and Anthropic latest models',
          },
        ],
      },
    ],
    plugins: [{ id: 'web', max_results: 5 }],
    max_output_tokens: 9000,
  }),
});

const result = await response.json();
console.log(result);
```

```python title="Python"
import requests

response = requests.post(
    'https://openrouter.ai/api/v1/responses',
    headers={
        'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
        'Content-Type': 'application/json',
    },
    json={
        'model': 'openai/o4-mini',
        'input': [
            {
                'type': 'message',
                'role': 'user',
                'content': [
                    {
                        'type': 'input_text',
                        'text': 'Compare OpenAI and Anthropic latest models',
                    },
                ],
            },
        ],
        'plugins': [{'id': 'web', 'max_results': 5}],
        'max_output_tokens': 9000,
    }
)

result = response.json()
print(result)
```

</CodeGroup>

## Web Search in Conversation

Include web search in multi-turn conversations:

<CodeGroup>

```typescript title="TypeScript"
const response = await fetch('https://openrouter.ai/api/v1/responses', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'openai/o4-mini',
    input: [
      {
        type: 'message',
        role: 'user',
        content: [
          {
            type: 'input_text',
            text: 'What is the latest version of React?',
          },
        ],
      },
      {
        type: 'message',
        id: 'msg_1',
        status: 'in_progress',
        role: 'assistant',
        content: [
          {
            type: 'output_text',
            text: 'Let me search for the latest React version.',
            annotations: [],
          },
        ],
      },
      {
        type: 'message',
        role: 'user',
        content: [
          {
            type: 'input_text',
            text: 'Yes, please find the most recent information',
          },
        ],
      },
    ],
    plugins: [{ id: 'web', max_results: 2 }],
    max_output_tokens: 9000,
  }),
});

const result = await response.json();
console.log(result);
```

```python title="Python"
import requests

response = requests.post(
    'https://openrouter.ai/api/v1/responses',
    headers={
        'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
        'Content-Type': 'application/json',
    },
    json={
        'model': 'openai/o4-mini',
        'input': [
            {
                'type': 'message',
                'role': 'user',
                'content': [
                    {
                        'type': 'input_text',
                        'text': 'What is the latest version of React?',
                    },
                ],
            },
            {
                'type': 'message',
                'id': 'msg_1',
                'status': 'in_progress',
                'role': 'assistant',
                'content': [
                    {
                        'type': 'output_text',
                        'text': 'Let me search for the latest React version.',
                        'annotations': [],
                    },
                ],
            },
            {
                'type': 'message',
                'role': 'user',
                'content': [
                    {
                        'type': 'input_text',
                        'text': 'Yes, please find the most recent information',
                    },
                ],
            },
        ],
        'plugins': [{'id': 'web', 'max_results': 2}],
        'max_output_tokens': 9000,
    }
)

result = response.json()
print(result)
```

</CodeGroup>

## Streaming Web Search

Monitor web search progress with streaming:

<CodeGroup>

```typescript title="TypeScript"
const response = await fetch('https://openrouter.ai/api/v1/responses', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'openai/o4-mini',
    input: [
      {
        type: 'message',
        role: 'user',
        content: [
          {
            type: 'input_text',
            text: 'What is the latest news about AI?',
          },
        ],
      },
    ],
    plugins: [{ id: 'web', max_results: 2 }],
    stream: true,
    max_output_tokens: 9000,
  }),
});

const reader = response.body?.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');

  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = line.slice(6);
      if (data === '[DONE]') return;

      try {
        const parsed = JSON.parse(data);
        if (parsed.type === 'response.output_item.added' &&
            parsed.item?.type === 'message') {
          console.log('Message added');
        }
        if (parsed.type === 'response.completed') {
          const annotations = parsed.response?.output
            ?.find(o => o.type === 'message')
            ?.content?.find(c => c.type === 'output_text')
            ?.annotations || [];
          console.log('Citations:', annotations.length);
        }
      } catch (e) {
        // Skip invalid JSON
      }
    }
  }
}
```

```python title="Python"
import requests
import json

response = requests.post(
    'https://openrouter.ai/api/v1/responses',
    headers={
        'Authorization': 'Bearer YOUR_OPENROUTER_API_KEY',
        'Content-Type': 'application/json',
    },
    json={
        'model': 'openai/o4-mini',
        'input': [
            {
                'type': 'message',
                'role': 'user',
                'content': [
                    {
                        'type': 'input_text',
                        'text': 'What is the latest news about AI?',
                    },
                ],
            },
        ],
        'plugins': [{'id': 'web', 'max_results': 2}],
        'stream': True,
        'max_output_tokens': 9000,
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        line_str = line.decode('utf-8')
        if line_str.startswith('data: '):
            data = line_str[6:]
            if data == '[DONE]':
                break
            try:
                parsed = json.loads(data)
                if (parsed.get('type') == 'response.output_item.added' and
                    parsed.get('item', {}).get('type') == 'message'):
                    print('Message added')
                if parsed.get('type') == 'response.completed':
                    output = parsed.get('response', {}).get('output', [])
                    message = next((o for o in output if o.get('type') == 'message'), {})
                    content = message.get('content', [])
                    text_content = next((c for c in content if c.get('type') == 'output_text'), {})
                    annotations = text_content.get('annotations', [])
                    print(f'Citations: {len(annotations)}')
            except json.JSONDecodeError:
                continue
```

</CodeGroup>

## Annotation Processing

Extract and process citation information:

<CodeGroup>

```typescript title="TypeScript"
function extractCitations(response: any) {
  const messageOutput = response.output?.find((o: any) => o.type === 'message');
  const textContent = messageOutput?.content?.find((c: any) => c.type === 'output_text');
  const annotations = textContent?.annotations || [];

  return annotations
    .filter((annotation: any) => annotation.type === 'url_citation')
    .map((annotation: any) => ({
      url: annotation.url,
      text: textContent.text.slice(annotation.start_index, annotation.end_index),
      startIndex: annotation.start_index,
      endIndex: annotation.end_index,
    }));
}

const result = await response.json();
const citations = extractCitations(result);
console.log('Found citations:', citations);
```

```python title="Python"
def extract_citations(response_data):
    output = response_data.get('output', [])
    message_output = next((o for o in output if o.get('type') == 'message'), {})
    content = message_output.get('content', [])
    text_content = next((c for c in content if c.get('type') == 'output_text'), {})
    annotations = text_content.get('annotations', [])
    text = text_content.get('text', '')

    citations = []
    for annotation in annotations:
        if annotation.get('type') == 'url_citation':
            citations.append({
                'url': annotation.get('url'),
                'text': text[annotation.get('start_index', 0):annotation.get('end_index', 0)],
                'start_index': annotation.get('start_index'),
                'end_index': annotation.get('end_index'),
            })

    return citations

result = response.json()
citations = extract_citations(result)
print(f'Found citations: {citations}')
```

</CodeGroup>

## Best Practices

1. **Limit results**: Use appropriate `max_results` to balance quality and speed
2. **Handle annotations**: Process citation annotations for proper attribution
3. **Query specificity**: Make search queries specific for better results
4. **Error handling**: Handle cases where web search might fail
5. **Rate limits**: Be mindful of search rate limits


## Next Steps

- Learn about [Tool Calling](./tool-calling) integration
- Explore [Reasoning](./reasoning) capabilities
- Review [Basic Usage](./basic-usage) fundamentals
---
title: Error Handling
subtitle: Understanding and handling errors in the Responses API Beta
headline: Responses API Beta Error Handling | Basic Error Guide
canonical-url: 'https://openrouter.ai/docs/api/reference/responses/error-handling'
'og:site_name': OpenRouter Documentation
'og:title': Responses API Beta Error Handling - Basic Error Guide
'og:description': >-
  Learn how to handle errors in OpenRouter's Responses API Beta with the basic
  error response format.
'og:image':
  type: url
  value: >-
    https://openrouter.ai/dynamic-og?title=Responses%20API%20Error%20Handling&description=Basic%20error%20handling%20guide
'og:image:width': 1200
'og:image:height': 630
'twitter:card': summary_large_image
'twitter:site': '@OpenRouterAI'
noindex: false
nofollow: false
---

<Warning title="Beta API">
  This API is in **beta stage** and may have breaking changes. Use with caution in production environments.
</Warning>

<Info title="Stateless Only">
  This API is **stateless** - each request is independent and no conversation state is persisted between requests. You must include the full conversation history in each request.
</Info>

The Responses API Beta returns structured error responses that follow a consistent format.

## Error Response Format

All errors follow this structure:

```json
{
  "error": {
    "code": "invalid_prompt",
    "message": "Detailed error description"
  },
  "metadata": null
}
```

### Error Codes

The API uses the following error codes:

| Code | Description | Equivalent HTTP Status |
|------|-------------|-------------|
| `invalid_prompt` | Request validation failed | 400 |
| `rate_limit_exceeded` | Too many requests | 429 |
| `server_error` | Internal server error | 500+ |
