# API 配置说明

## 重要提醒
为了保护您的API密钥安全，本项目已将敏感配置文件添加到`.gitignore`中，防止意外提交到版本控制系统。

## 配置步骤

### 1. 复制配置模板
```bash
cp config_template.py config.py
```

### 2. 编辑配置文件
打开 `config.py` 文件，将以下占位符替换为您的实际API密钥：

```python
# DeepSeek API 配置
DEEPSEEK_API_KEY = "your_deepseek_api_key_here"  # 替换为您的DeepSeek API密钥

# OpenAI API 配置
OPENAI_API_KEY = "your_openai_api_key_here"      # 替换为您的OpenAI API密钥
```

### 3. 环境变量方式（可选）
您也可以通过设置环境变量来配置API密钥：

```bash
# 在 ~/.bashrc 或 ~/.zshrc 中添加
export DEEPSEEK_API_KEY="your_deepseek_api_key_here"
export OPENAI_API_KEY="your_openai_api_key_here"
```

## 文件说明

- `config_template.py`: 配置文件模板，包含所有配置项的说明
- `config.py`: 实际配置文件（不会被提交到Git）
- `deepseek.py`: DeepSeek API调用脚本
- `test_gpt5.py`: OpenAI GPT-5 API测试脚本

## 安全注意事项

1. **永远不要**将包含真实API密钥的文件提交到版本控制系统
2. `config.py` 文件已被添加到 `.gitignore` 中
3. 如果意外提交了API密钥，请立即：
   - 撤销提交
   - 重新生成API密钥
   - 更新配置文件

## 获取API密钥

### DeepSeek API
1. 访问 [DeepSeek开放平台](https://platform.deepseek.com/)
2. 注册账号并登录
3. 在API密钥管理页面创建新的API密钥

### OpenAI API
1. 访问 [OpenAI平台](https://platform.openai.com/)
2. 注册账号并登录
3. 在API密钥页面创建新的API密钥

## 故障排除

如果遇到"未找到config.py文件"的警告：
1. 确保已经复制了配置模板：`cp config_template.py config.py`
2. 确保配置文件在正确的目录中
3. 检查配置文件中的API密钥是否正确填写

如果遇到API连接错误：
1. 检查网络连接
2. 验证API密钥是否有效
3. 确认API服务是否正常运行