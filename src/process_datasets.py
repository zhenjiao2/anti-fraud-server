"""
数据集处理脚本
使用agent.py程序处理所有数据集，提取rating值
"""

import asyncio
import pandas as pd
import json
import os
import sys
import argparse
from typing import List, Dict, Optional
import time

start_time = time.time()
# 导入工作流程序
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 修复文件名中的特殊字符
import importlib.util
spec = importlib.util.spec_from_file_location("agent_workflow", os.path.join(os.path.dirname(__file__), "agent.py"))
agent_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_module)
LangChainFraudWorkflow = agent_module.LangChainFraudWorkflow
load_config = agent_module.load_config

async def process_single_content(workflow: LangChainFraudWorkflow, content: str) -> int:
    """
    处理单条content内容，返回rating值
    """
    retry = 0
    while retry < 3:
        try:
            # 调用工作流
            result = await workflow.ainvoke({"text": content})
            
            # 从最终分析中提取rating
            final_analysis = result.get("final_analysis", "")
            
            # 尝试解析JSON格式的分析结果
            json_start = final_analysis.find('{')
            json_end = final_analysis.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = final_analysis[json_start:json_end]
                analysis_json = json.loads(json_str)
                rating = analysis_json.get("rating", 0)  # 默认值0
                if rating > 0:
                    return rating
            retry += 1
            print(f"警告: 分析失败，第{retry}次重试中...")
            continue
        except Exception as e:
            retry += 1
            print(f"警告: 第{retry}次重试出错，错误信息: {str(e)}")
            
    print(f"多次重试后仍无法提取rating值，使用默认值0")
    return 0

async def process_dataset(file_path: str, workflow: LangChainFraudWorkflow, model_name: str) -> None:
    """
    处理单个数据集文件
    """
    print(f"\n开始处理数据集: {file_path}")
    
    # 读取CSV文件
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='gbk')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin-1')
    
    print(f"数据集包含 {len(df)} 条记录")
    
    # 处理每条记录
    for index, row in df.iterrows():
        content = str(row['content'])
        
        print(f"处理第 {index + 1}/{len(df)} 条记录...")
        print(f"内容: {content[:100]}...")  # 显示前100个字符用于调试
        
        # 调用工作流分析
        rating = await process_single_content(workflow, content)
        
        # 更新rating列
        df.at[index, 'rating'] = rating
        
        print(f"第 {index + 1} 条记录完成，rating: {rating}")
        
        # 每处理10条记录保存一次
        if (index + 1) % 10 == 0:
            w_file_path = file_path.replace('.csv', f'_{model_name}_test.csv')
            df.to_csv(w_file_path, index=False, encoding='utf-8')
            print(f"已保存前 {index + 1} 条记录的结果")
    
    # 最终保存
    w_file_path = file_path.replace('.csv', f'_{model_name}_test.csv')
    df.to_csv(w_file_path, index=False, encoding='utf-8')
    print(f"数据集 {file_path} 处理完成并已保存")

async def main(config_path: Optional[str] = None, verbose_level: Optional[int] = None, dataset_file: Optional[str] = None):
    """
    主函数：处理所有数据集
    
    Args:
        config_path: 配置文件路径
        verbose_level: 详细程度级别
        dataset_file: 指定要处理的数据集文件
    """
    print("开始处理反欺诈数据集...")
    
    # 加载配置文件
    config = load_config(config_path)
    
    # 如果命令行指定了verbose级别，优先使用
    if verbose_level is not None:
        config.setdefault("workflow", {})["verbose_level"] = verbose_level
    
    # 初始化工作流，传入配置
    workflow = LangChainFraudWorkflow(config)
    
    # 从配置中获取模型名称，用于文件命名
    model_name = config.get("azure_ai_inference", {}).get("model_name", "unknown_model")
    print(f"使用模型: {model_name}")
    
    # 数据集文件列表
    if dataset_file:
        # 如果指定了特定文件，只处理该文件
        dataset_files = [dataset_file]
    else:
        # 否则使用默认文件列表
        dataset_files = [
            #r"datasets/dataset-客服.csv",
            #r"datasets/dataset-贷款.csv",
            #r"datasets/dataset-冒充熟人.csv",
            #r"datasets/dataset-公检法.csv",
            r"datasets/dataset-正常短信.csv",
            r"datasets/dataset-广告.csv"
        ]
    
    try:
        # 处理每个数据集
        for file_path in dataset_files:
            if os.path.exists(file_path):
                await process_dataset(file_path, workflow, model_name)
            else:
                print(f"文件不存在: {file_path}")
    
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
    
    finally:
        # 清理资源
        await workflow.cleanup()
        print("\n所有数据集处理完成！")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="反欺诈数据集处理脚本")
    parser.add_argument(
        "--config", 
        "-c", 
        type=str, 
        help="配置文件路径 (JSON格式)",
        default=None
    )
    parser.add_argument(
        "--verbose",
        "-v",
        type=int,
        choices=[0, 1, 2, 3],
        help="详细程度级别 (0=仅错误, 1=默认, 2=详细, 3=调试)",
        default=None
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        help="指定要处理的数据集文件路径",
        default=None
    )
    
    args = parser.parse_args()
    
    print("启动反欺诈数据集处理脚本...")
    if args.config:
        print(f"使用配置文件: {args.config}")
    if args.verbose is not None:
        print(f"详细程度级别: {args.verbose}")
    if args.dataset:
        print(f"指定数据集: {args.dataset}")
    
    # 运行异步主函数
    asyncio.run(main(config_path=args.config, verbose_level=args.verbose, dataset_file=args.dataset))
end_time = time.time()
print(end_time - start_time, "seconds")