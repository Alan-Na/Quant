#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Workflow Prototype
Generator (DeepSeek) -> Critic (ChatGPT) -> Editor (Local Merge)
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Union

import requests
from pydantic import BaseModel, Field, ValidationError, field_validator

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1/chat/completions"
OPENAI_BASE_URL = "https://api.openai.com/v1/chat/completions"


class RecommendationItem(BaseModel):
    symbol: str
    thesis_bullets: List[str] = Field(min_length=1, max_length=3)
    time_horizon: str
    confidence: Union[float, str]
    risks: List[str] = Field(min_length=1, max_length=3)
    evidence_links: List[str]
    model: str
    disclaimers: str

    @field_validator("time_horizon")
    @classmethod
    def validate_time_horizon(cls, value: str) -> str:
        allowed = {"short", "medium", "long"}
        if value not in allowed:
            raise ValueError("time_horizon must be one of short/medium/long")
        return value

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: Union[float, str]) -> Union[float, str]:
        if isinstance(value, str):
            allowed = {"low", "med", "high"}
            if value not in allowed:
                raise ValueError("confidence must be 0-1 or low/med/high")
            return value
        if not 0 <= value <= 1:
            raise ValueError("confidence must be 0-1")
        return value

    @field_validator("evidence_links")
    @classmethod
    def validate_evidence_links(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("evidence_links cannot be empty")
        return value


class CritiqueItem(BaseModel):
    symbol: str
    issues: List[str] = Field(min_length=1, max_length=3)
    evidence_links: List[str]
    uncertainty: str


class GeneratorOutput(BaseModel):
    recommendations: List[RecommendationItem]


class CriticOutput(BaseModel):
    critiques: List[CritiqueItem]


@dataclass
class ModelResponse:
    content: str
    model: str


def call_api(url: str, api_key: str, model: str, messages: List[dict]) -> ModelResponse:
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "temperature": 0.7, "max_tokens": 2000},
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return ModelResponse(content=data["choices"][0]["message"]["content"], model=model)


def build_generator_prompt(topic: str, count: int) -> str:
    return f"""
You are the Generator. Provide {count} candidates for the topic \"{topic}\".
Return strictly valid JSON with this structure:
{{
  "recommendations": [
    {{
      "symbol": "...",
      "thesis_bullets": ["...", "..."],
      "time_horizon": "short/medium/long",
      "confidence": 0.0-1.0 or "low/med/high",
      "risks": ["..."],
      "evidence_links": ["N/A"],
      "model": "deepseek",
      "disclaimers": "..."
    }}
  ]
}}
Output JSON only. No extra text.
""".strip()


def build_critic_prompt(topic: str, generator_json: str) -> str:
    return f"""
You are the Critic. Identify gaps, missing evidence, and uncertainties in the Generator output.
Return strictly valid JSON:
{{
  "critiques": [
    {{
      "symbol": "...",
      "issues": ["..."],
      "evidence_links": ["N/A"],
      "uncertainty": "..."
    }}
  ]
}}
Output JSON only. No extra text.

Topic: {topic}
Generator output:
{generator_json}
""".strip()


def parse_with_retry(model_fn, prompt: str, schema, max_attempts: int = 2) -> BaseModel:
    last_error = None
    for attempt in range(1, max_attempts + 1):
        response = model_fn(prompt)
        try:
            return schema.model_validate_json(response.content)
        except ValidationError as exc:
            last_error = exc
            if attempt == max_attempts:
                break
    raise ValueError(f"Schema validation failed: {last_error}")


def merge_outputs(generator: GeneratorOutput, critic: CriticOutput) -> dict:
    critique_map = {item.symbol: item for item in critic.critiques}
    merged = []
    for rec in generator.recommendations:
        critique = critique_map.get(rec.symbol)
        conflict = bool(critique and critique.issues)
        merged.append(
            {
                **rec.model_dump(),
                "critic_issues": critique.issues if critique else [],
                "critic_uncertainty": critique.uncertainty if critique else "",
                "conflict": conflict,
            }
        )
    return {
        "recommendations": merged,
        "summary": {
            "total": len(merged),
            "conflicts": sum(1 for item in merged if item["conflict"]),
        },
    }


def load_api_keys() -> tuple[str, str]:
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config_keys = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config_keys = json.load(f)
    deepseek_key = os.getenv("DEEPSEEK_API_KEY") or config_keys.get("DEEPSEEK_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY") or config_keys.get("OPENAI_API_KEY")
    return deepseek_key, openai_key


def main() -> int:
    parser = argparse.ArgumentParser(description="AI Workflow Prototype")
    parser.add_argument("--topic", default="Market watchlist")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--deepseek-model", default="deepseek-reasoner")
    parser.add_argument("--openai-model", default="gpt-5")
    parser.add_argument("--output-json", default="output.json")
    parser.add_argument("--output-md", default="output.md")
    args = parser.parse_args()

    deepseek_key, openai_key = load_api_keys()
    if not deepseek_key or not openai_key:
        raise ValueError("Missing DEEPSEEK_API_KEY or OPENAI_API_KEY")

    def generator_call(prompt: str) -> ModelResponse:
        return call_api(
            DEEPSEEK_BASE_URL,
            deepseek_key,
            args.deepseek_model,
            messages=[{"role": "user", "content": prompt}],
        )

    def critic_call(prompt: str) -> ModelResponse:
        return call_api(
            OPENAI_BASE_URL,
            openai_key,
            args.openai_model,
            messages=[{"role": "user", "content": prompt}],
        )

    generator_prompt = build_generator_prompt(args.topic, args.count)
    generator_output = parse_with_retry(generator_call, generator_prompt, GeneratorOutput)

    critic_prompt = build_critic_prompt(args.topic, generator_output.model_dump_json())
    critic_output = parse_with_retry(critic_call, critic_prompt, CriticOutput)

    final_report = merge_outputs(generator_output, critic_output)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("# AI Workflow Report\n\n")
        f.write(f"Topic: {args.topic}\n\n")
        for idx, item in enumerate(final_report["recommendations"], start=1):
            f.write(f"## {idx}. {item['symbol']}\n")
            f.write(f"- Thesis: {', '.join(item['thesis_bullets'])}\n")
            f.write(f"- Time Horizon: {item['time_horizon']}\n")
            f.write(f"- Confidence: {item['confidence']}\n")
            f.write(f"- Risks: {', '.join(item['risks'])}\n")
            f.write(f"- Evidence: {', '.join(item['evidence_links'])}\n")
            if item["conflict"]:
                f.write(f"- Conflict: Yes\n")
                f.write(f"- Critic Issues: {', '.join(item['critic_issues'])}\n")
            f.write("\n")

    print(f"Saved {args.output_json} and {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
