#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件模板
请复制此文件为 config.py 并填入您的实际API密钥
注意：config.py 已被添加到 .gitignore 中，不会被提交到版本控制
"""

# DeepSeek API 配置
DEEPSEEK_API_KEY = "your_deepseek_api_key_here"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# OpenAI API 配置
OPENAI_API_KEY = "your_openai_api_key_here"
OPENAI_BASE_URL = "https://api.openai.com/v1"

# 其他配置
DEFAULT_MODEL = "gpt-5"
MAX_OUTPUT_TOKENS = 6000
REASONING_EFFORT = "medium"

# 日志配置
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# 报告配置
REPORT_DIR = "reports"
DATA_CACHE_DIR = "data_cache"