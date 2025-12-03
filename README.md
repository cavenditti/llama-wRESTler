<p align="center">
  <img src="docs/banner_cropped.png" alt="Llama wRESTler Banner" width="100%">
</p>

# 🦙 Llama wRESTler

**Automated REST API testing powered by LLMs**

`llama-wrestler` is an intelligent API testing tool that automatically generates and executes test plans for REST APIs using OpenAI-compatible LLMs. Point it at an OpenAPI specification, and it will analyze your API, generate meaningful test cases, create realistic test data, and execute the tests—all autonomously.

Think of it like a [RESTler fuzzer](https://github.com/microsoft/restler-fuzzer) but with more llamas 🦙.

> [!CAUTION]
>
> Current state: I'm still brainstorming ideas about what direction this should take.
> It is somewhat usable but I'm experimenting and iterating very quickly (with a lot of vibe coding along the road).

## ✨ Features

- **Automatic Test Plan Generation**: Analyzes your OpenAPI spec and creates a comprehensive test plan with proper endpoint ordering and dependencies
- **Intelligent Test Data Generation**: Uses LLMs to generate realistic, contextually appropriate test data
- **Dependency Resolution**: Automatically orders tests based on dependencies (e.g., create before update/delete)
- **Optional Repository Analysis**: Can analyze your source code for additional context
- **Detailed Execution Reports**: Saves test plans, generated data, and execution results as JSON
- **Authentication Support**: Handles OAuth2 authentication nicely

### Planned features in the future

- Plan and mock data caching/reusing
- bulk semi-random data generation to fuzz endpoints
- Custom extra instructions for LLMs
- Pluggable custom authentication schemes?
- Add the missing tests, to ensure you're testing with tested code

## 🚀 Quick Start

### Installation

```bash
# Run directly with uvx (recommended)
uvx llama-wrestler https://petstore.swagger.io/v2/swagger.json

# Or install with pip/uv
pip install llama-wrestler
# or
uv add llama-wrestler
```

### Configuration

Create a `.env` file in your working directory with your LLM configuration:

```bash
# Model configuration using pydantic-ai format: "provider:model-name"
# Primary model for test plan generation and refinement (complex reasoning tasks)
MODEL=openai:gpt-4o

# Weaker/cheaper model for summarization tasks (faster, cheaper)
WEAK_MODEL=openai:gpt-4o-mini
```

#### Supported Providers

| Provider | Model Format | API Key Variable |
|----------|--------------|------------------|
| OpenAI | `openai:gpt-4o` | `OPENAI_API_KEY` |
| Anthropic | `anthropic:claude-sonnet-4-5` | `ANTHROPIC_API_KEY` |
| Google | `google:gemini-2.5-pro` | `GOOGLE_API_KEY` |
| Groq | `groq:llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| Mistral | `mistral:mistral-large-latest` | `MISTRAL_API_KEY` |
| Together AI | `together:meta-llama/Llama-3.3-70B-Instruct-Turbo-Free` | `TOGETHER_API_KEY` |
| Hugging Face | `huggingface:Qwen/Qwen3-235B-A22B` | `HUGGINGFACE_API_KEY` |
| Cerebras | `cerebras:llama3.3-70b` | `CEREBRAS_API_KEY` |
| Fireworks | `fireworks:accounts/fireworks/models/qwq-32b` | `FIREWORKS_API_KEY` |

#### Example Configurations

```bash
# OpenAI (default)
MODEL=openai:gpt-4o
WEAK_MODEL=openai:gpt-4o-mini
OPENAI_API_KEY=sk-...

# Anthropic Claude
MODEL=anthropic:claude-sonnet-4-5
WEAK_MODEL=anthropic:claude-3-5-haiku-latest
ANTHROPIC_API_KEY=sk-ant-...

# Google Gemini
MODEL=google:gemini-2.5-pro
WEAK_MODEL=google:gemini-2.0-flash
GOOGLE_API_KEY=...

# Mixed providers (use the best of each!)
MODEL=anthropic:claude-sonnet-4-5
WEAK_MODEL=openai:gpt-4o-mini
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Groq (fast inference)
MODEL=groq:llama-3.3-70b-versatile
WEAK_MODEL=groq:llama-3.1-8b-instant
GROQ_API_KEY=gsk_...

# OpenRouter (access 100+ models via one API)
MODEL=openrouter:anthropic/claude-sonnet-4
WEAK_MODEL=openrouter:openai/gpt-4o-mini
# OPENAI_API_KEY won't be used for OpenRouter models
# Please set OPENROUTER_API_KEY in your environment if needed
OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

#### Legacy Configuration (backwards compatible)

The old OpenAI-specific settings still work:

```bash
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4o
OPENAI_WEAK_MODEL=gpt-4o-mini
```

## 📖 Usage

### Basic Usage

```bash
# Test an API from its OpenAPI spec URL
uvx llama-wrestler https://api.example.com/openapi.json
```

### With Authentication

```bash
# Using command-line credentials
uvx llama-wrestler https://api.example.com/openapi.json \
  --username user@example.com \
  --password secretpassword

# Using a credentials file
uvx llama-wrestler https://api.example.com/openapi.json \
  --credentials-file credentials.json
```

Credentials file format (`credentials.json`):

```json
{
  "username": "your_test_user@example.com",
  "password": "your_test_password",
  "extra": {
    "grant_type": "password"
  }
}
```

### With Repository Analysis

Provide a local repository path for additional context during test generation:

```bash
uvx llama-wrestler https://api.example.com/openapi.json --repo /path/to/api/repo
```

### Skip Test Execution

Generate the test plan and test data without running the tests:

```bash
uvx llama-wrestler https://api.example.com/openapi.json --skip-execution
```

### Command-Line Options

| Option | Short | Description |
|--------|-------|-------------|
| `openapi_url` | - | URL to the OpenAPI specification (required) |
| `--repo` | - | Path to local repository for analysis |
| `--skip-execution` | - | Skip the test execution phase |
| `--username` | `-u` | Username/email for authentication |
| `--password` | `-p` | Password for authentication |
| `--credentials-file` | `-c` | Path to JSON credentials file |

## 🔄 How It Works

Llama wRESTler operates in three phases:

### Phase 1: Preliminary Analysis

- Fetches and parses the OpenAPI specification
- Optionally analyzes the source repository
- Generates a test plan with ordered steps and dependencies (basically an ordered graph of the requests to send)

### Phase 2: Test Data Generation

- Creates realistic test data for each endpoint
- Generates appropriate payloads based on schemas
- Prepares authentication flows

### Phase 3: Test Execution

- Executes tests in dependency order
- Manages authentication tokens
- Captures and reports results

## 📁 Output Structure

Results are saved in the `output/` directory with the following structure:

```
output/
└── 001_api_example_com_20231202_143052/
    ├── openapi_spec.json      # The fetched OpenAPI specification
    ├── test_plan.json         # Generated test plan with steps
    ├── test_data.json         # Generated test data for each step
    └── execution_results.json # Test execution results
```

## 🔧 Using with Local/Custom LLMs

Llama wRESTler works with any OpenAI-compatible API through pydantic-ai. For local providers like Ollama, you can use the OpenAI-compatible interface:

```bash
# .env for Ollama (via OpenAI-compatible endpoint)
MODEL=openai:llama3.2
WEAK_MODEL=openai:llama3.2
OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://localhost:11434/v1
```

For LiteLLM proxy or other OpenAI-compatible servers, configure the base URL accordingly.

## 📋 Requirements

- Python 3.10+
- An OpenAI-compatible API endpoint

## 📄 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🫂 Acknowledgements

This idea spun out while working with the friends at [LAIF](https://laifgroup.com/).
