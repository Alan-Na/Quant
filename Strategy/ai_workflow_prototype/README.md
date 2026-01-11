# AI Workflow Prototype (Generator → Critic → Editor)

This folder contains a minimal, three-step AI workflow prototype:

1. **Generator**: DeepSeek produces structured candidates.
2. **Critic**: ChatGPT validates, challenges, and flags uncertainties.
3. **Editor**: Local code merges both outputs into a final report (conflicts, consensus, ranking).

The goal is to demonstrate a rapid, AI-enhanced workflow with structured JSON output and validation.

## Requirements

- Python 3.10+
- `requests`
- `pydantic`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

This script reads API keys from environment variables or a local `config.json` file:

```bash
export DEEPSEEK_API_KEY="..."
export OPENAI_API_KEY="..."
```

Optional `config.json` (copy from `config_template.json`):

```json
{
  "DEEPSEEK_API_KEY": "your_deepseek_api_key_here",
  "OPENAI_API_KEY": "your_openai_api_key_here"
}
```

## Run

```bash
python workflow.py --topic "Market watchlist" --count 5
```

Output:
- `output.json` (structured results)
- `output.md` (final report)

## Notes

- If schema validation fails, the script automatically re-prompts once.
- Evidence links are optional; use `"N/A"` when not available.
- This is a minimal PoC intended for internship applications.
