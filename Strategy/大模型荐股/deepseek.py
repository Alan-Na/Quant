#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek股票推荐策略
基于DeepSeek API的智能股票推荐系统
每日自动生成推荐报告
"""

import requests
import json
import os
from datetime import datetime, timedelta
import re
from typing import List, Dict, Optional
import logging
from dataclasses import dataclass
try:
    from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
except ImportError:
    print("警告: 未找到config.py文件，请从config_template.py复制并配置您的API密钥")
    DEEPSEEK_API_KEY = None
    DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class StockRecommendation:
    """股票推荐数据结构"""
    symbol: str
    name: str
    reason: str
    confidence: float
    target_price: Optional[float] = None
    risk_level: str = "中等"

class DeepSeekStockAdvisor:
    """DeepSeek股票推荐顾问"""
    
    def __init__(self, api_key: str = None):
        """
        初始化DeepSeek股票推荐顾问
        
        Args:
            api_key: DeepSeek API密钥，如果为None则从环境变量获取
        """
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY') or DEEPSEEK_API_KEY
        if not self.api_key:
            raise ValueError("请设置DEEPSEEK_API_KEY环境变量、配置config.py文件或传入api_key参数")
        
        self.base_url = DEEPSEEK_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 报告保存目录
        self.reports_dir = os.path.join(os.path.dirname(__file__), "reports")
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def generate_daily_prompt(self, date: datetime = None) -> str:
        """
        生成每日股票推荐提示词
        
        Args:
            date: 目标日期，默认为今天
            
        Returns:
            格式化的提示词
        """
        if date is None:
            date = datetime.now()
        
        # 获取前一个交易日（简化处理，实际应考虑节假日）
        prev_date = date - timedelta(days=1)
        if date.weekday() == 0:  # 周一
            prev_date = date - timedelta(days=3)
        
        prompt = f"""
你是一位专业的股票投资顾问，具有丰富的A股市场分析经验。请基于以下要求为我推荐今日（{date.strftime('%Y年%m月%d日')}）值得关注的A股股票，非科创板：

## 分析要求：
1. 请结合当前经济形势、市场环境、政策导向、行业趋势等因素
2. 重点关注具有成长潜力的优质公司
3. 考虑技术面和基本面的综合分析
4. 注意风险控制，避免推荐高风险标的

## 推荐格式：
请严格按照以下JSON格式输出，推荐5-7只股票：

```json
{{
  "date": "{date.strftime('%Y-%m-%d')}",
  "market_outlook": "今日市场整体展望（50字以内）",
  "recommendations": [
    {{
      "symbol": "股票代码（如000001）",
      "name": "股票名称",
      "reason": "推荐理由（100字以内）",
      "confidence": 0.85,
      "target_price": 12.50,
      "risk_level": "低/中等/高"
    }}
  ],
  "risk_warning": "风险提示（50字以内）"
}}
```

## 注意事项：
- 股票代码必须是真实存在的A股代码
- 置信度范围：0.1-1.0
- 目标价格要合理，基于当前价格给出
- 推荐理由要具体，避免空泛表述
- 若数据不足或不满足风险约束，宁可减少推荐也不要臆断
- 必须包含风险提示

请开始你的分析和推荐：
        """.strip()
        
        return prompt
    
    def call_deepseek_api(self, prompt: str, model: str = "deepseek-reasoner", enable_search: bool = True) -> Dict:
        """
        调用DeepSeek API
        
        Args:
            prompt: 输入提示词
            model: 使用的模型名称 (deepseek-chat 或 deepseek-reasoner)
            enable_search: 是否启用联网搜索功能
            
        Returns:
            包含API响应内容和推理过程的字典
        """
        # 构建消息，添加联网搜索指令
        messages = []
        if enable_search:
            messages.append({
                "role": "system",
                "content": "你是一位专业的股票投资顾问。请使用联网搜索功能获取最新的市场信息、政策动态、行业新闻等，以提供更准确和及时的股票推荐。"
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 1.0,
            "max_tokens": 4000  # R1模型支持更长的输出
        }
        
        try:
            logger.info("正在调用DeepSeek API...")
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=300  # R1模型可能需要更长时间
            )
            
            response.raise_for_status()
            result = response.json()
            
            # 提取主要内容和推理过程
            choice = result['choices'][0]
            content = choice['message']['content']
            
            # 如果是R1模型，可能包含推理过程
            reasoning = None
            if model == "deepseek-reasoner" and 'reasoning_content' in choice['message']:
                reasoning = choice['message']['reasoning_content']
            
            logger.info("API调用成功")
            return {
                'content': content,
                'reasoning': reasoning,
                'model': model,
                'usage': result.get('usage', {})
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API调用失败: {e}")
            raise Exception(f"API调用失败: {str(e)}")
        except KeyError as e:
            logger.error(f"API响应格式错误: {e}")
            raise Exception(f"API响应格式错误: {str(e)}")
    
    def parse_recommendations(self, api_response: str) -> Dict:
        """
        解析API响应，提取股票推荐信息
        
        Args:
            api_response: API返回的原始文本
            
        Returns:
            解析后的推荐数据
        """
        try:
            # 尝试提取JSON部分
            json_match = re.search(r'```json\s*({.*?})\s*```', api_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 如果没有找到代码块，尝试直接解析
                json_str = api_response.strip()
            
            # 解析JSON
            data = json.loads(json_str)
            
            # 验证必要字段
            required_fields = ['date', 'market_outlook', 'recommendations', 'risk_warning']
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"缺少必要字段: {field}")
            
            # 验证推荐列表
            recommendations = data['recommendations']
            if not isinstance(recommendations, list) or len(recommendations) == 0:
                raise ValueError("推荐列表为空或格式错误")
            
            # 验证每个推荐的字段
            for i, rec in enumerate(recommendations):
                required_rec_fields = ['symbol', 'name', 'reason', 'confidence']
                for field in required_rec_fields:
                    if field not in rec:
                        raise ValueError(f"推荐{i+1}缺少字段: {field}")
            
            logger.info(f"成功解析{len(recommendations)}个股票推荐")
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            logger.error(f"原始响应: {api_response}")
            raise ValueError("API响应不是有效的JSON格式")
        except Exception as e:
            logger.error(f"解析推荐数据失败: {e}")
            raise
    
    def generate_summary_report(self, recommendations_data: Dict, api_result: Dict = None) -> str:
        """
        生成推荐总结报告
        
        Args:
            recommendations_data: 解析后的推荐数据
            api_result: API调用结果，包含推理过程等信息
            
        Returns:
            格式化的报告文本
        """
        date = recommendations_data['date']
        market_outlook = recommendations_data['market_outlook']
        recommendations = recommendations_data['recommendations']
        risk_warning = recommendations_data['risk_warning']
        
        # 获取模型信息
        model_info = ""
        if api_result and isinstance(api_result, dict):
            model_name = api_result.get('model', 'deepseek-chat')
            usage = api_result.get('usage', {})
            model_info = f"\n**使用模型**: {model_name}"
            if usage:
                model_info += f"\n**Token使用**: 输入{usage.get('prompt_tokens', 0)}, 输出{usage.get('completion_tokens', 0)}"
            
        report = f"""
# DeepSeek股票推荐报告

**日期**: {date}
**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{model_info}
**数据来源**: DeepSeek AI + 实时联网搜索

## 市场展望
{market_outlook}

## 推荐股票

"""
        
        for i, rec in enumerate(recommendations, 1):
            confidence_stars = "★" * int(rec['confidence'] * 5)
            target_price_str = f"目标价: {rec.get('target_price', 'N/A')}" if rec.get('target_price') else ""
            
            report += f"""
### {i}. {rec['name']} ({rec['symbol']})

- **推荐理由**: {rec['reason']}
- **置信度**: {rec['confidence']:.2f} {confidence_stars}
- **风险等级**: {rec.get('risk_level', '中等')}
{f"- **{target_price_str}**" if target_price_str else ""}

"""
        
        # 添加AI推理过程（如果有）
        if api_result and api_result.get('reasoning'):
            report += f"""
## 🧠 AI推理过程

{api_result['reasoning']}

"""
        
        report += f"""
## 风险提示
{risk_warning}

---
*本报告由DeepSeek AI生成，结合实时联网搜索数据，仅供参考，不构成投资建议。投资有风险，入市需谨慎。*
        """
        
        return report.strip()
    
    def save_report(self, report: str, date: datetime = None) -> str:
        """
        保存报告到文件
        
        Args:
            report: 报告内容
            date: 报告日期
            
        Returns:
            保存的文件路径
        """
        if date is None:
            date = datetime.now()
        
        filename = f"deepseek_stock_report_{date.strftime('%Y%m%d')}.md"
        filepath = os.path.join(self.reports_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"报告已保存到: {filepath}")
        return filepath
    
    def run_daily_recommendation(self, date: datetime = None) -> Dict:
        """
        运行每日推荐流程
        
        Args:
            date: 目标日期，默认为今天
            
        Returns:
            包含推荐数据和报告路径的字典
        """
        if date is None:
            date = datetime.now()
        
        logger.info(f"开始生成{date.strftime('%Y-%m-%d')}的股票推荐")
        
        try:
            # 1. 生成提示词
            prompt = self.generate_daily_prompt(date)
            logger.info("提示词生成完成")
            
            # 2. 调用API
            api_result = self.call_deepseek_api(prompt)
            api_response = api_result['content'] if isinstance(api_result, dict) else api_result
            
            # 3. 解析推荐
            recommendations_data = self.parse_recommendations(api_response)
            
            # 4. 生成报告
            report = self.generate_summary_report(recommendations_data, api_result)
            
            # 5. 保存报告
            report_path = self.save_report(report, date)
            
            logger.info("每日推荐流程完成")
            
            return {
                'success': True,
                'date': date.strftime('%Y-%m-%d'),
                'recommendations': recommendations_data,
                'report_path': report_path,
                'api_result': api_result,
                'summary': f"成功生成{len(recommendations_data['recommendations'])}个股票推荐"
            }
            
        except Exception as e:
            logger.error(f"每日推荐流程失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'date': date.strftime('%Y-%m-%d')
            }

def main():
    """
    主函数 - 命令行入口
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='DeepSeek股票推荐系统')
    parser.add_argument('--api-key', help='DeepSeek API密钥')
    parser.add_argument('--date', help='目标日期 (YYYY-MM-DD格式)', default=None)
    parser.add_argument('--model', type=str, default='deepseek-reasoner', 
                       choices=['deepseek-chat', 'deepseek-reasoner'],
                       help='选择模型 (默认: deepseek-reasoner)')
    parser.add_argument('--no-search', action='store_true', 
                       help='禁用联网搜索功能')
    
    args = parser.parse_args()
    
    try:
        # 解析日期
        target_date = None
        if args.date:
            target_date = datetime.strptime(args.date, '%Y-%m-%d')
        
        # 创建顾问实例
        advisor = DeepSeekStockAdvisor(api_key=args.api_key)
        
        # 临时修改默认模型和搜索设置
        original_call = advisor.call_deepseek_api
        def custom_call(prompt, model=args.model, enable_search=not args.no_search):
            return original_call(prompt, model, enable_search)
        advisor.call_deepseek_api = custom_call
        
        # 运行推荐
        result = advisor.run_daily_recommendation(target_date)
        
        if result['success']:
            print(f"✅ 推荐生成成功!")
            print(f"📅 日期: {result['date']}")
            print(f"📊 {result['summary']}")
            print(f"📄 报告路径: {result['report_path']}")
            
            # 显示模型使用信息
            api_result = result.get('api_result', {})
            if api_result:
                print(f"🤖 使用模型: {api_result.get('model', 'N/A')}")
                print(f"🌐 联网搜索: {'启用' if not args.no_search else '禁用'}")
                
                usage = api_result.get('usage', {})
                if usage:
                    print(f"💰 Token使用: 输入{usage.get('prompt_tokens', 0)}, 输出{usage.get('completion_tokens', 0)}")
                    
                if api_result.get('reasoning'):
                    print(f"🧠 包含推理过程: {len(api_result['reasoning'])} 字符")
            
            # 打印推荐概要
            recommendations = result['recommendations']['recommendations']
            print("\n📈 推荐股票:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec['name']} ({rec['symbol']}) - 置信度: {rec['confidence']:.2f}")
        else:
            print(f"❌ 推荐生成失败: {result['error']}")
            
    except Exception as e:
        print(f"❌ 程序运行失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())