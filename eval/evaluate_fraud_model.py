"""
反欺诈评估脚本 - 预测rating vs 人工标注human_rating的全面评估
专门针对反欺诈场景设计的评估指标

主指标：
- QWK (Quadratic Weighted Kappa)
- MAE (Mean Absolute Error)  
- Weighted F1 / Macro F1

辅助指标：
- Recall@HighRisk (对真实=4/5的召回率)
- FPR@LowRisk (真实=1/2的误判率)
- Cost-aware Error (成本加权误差)
- Confusion Matrix (特别关注错位方向)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    cohen_kappa_score, 
    mean_absolute_error, 
    f1_score, 
    confusion_matrix,
    classification_report,
    recall_score,
    precision_score
)
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess_data(file_path: str):
    """加载和预处理数据"""
    print("=== 数据加载和预处理 ===")
    
    # 读取数据
    df = pd.read_csv(file_path, encoding='utf-8')
    print(f"原始数据量: {len(df)} 条")
    
    # 检查列名
    print(f"列名: {df.columns.tolist()}")
    
    # 数据清洗 - 移除缺失值
    original_len = len(df)
    df = df.dropna(subset=['human rating', 'rating'])
    print(f"移除缺失值后: {len(df)} 条 (移除了 {original_len - len(df)} 条)")
    
    # 确保评分在1-5范围内
    df = df[(df['human rating'] >= 1) & (df['human rating'] <= 5)]
    df = df[(df['rating'] >= 1) & (df['rating'] <= 5)]
    print(f"筛选1-5评分后: {len(df)} 条")
    
    # 转换为整数（四舍五入）
    df['human_rating_int'] = df['human rating'].round().astype(int)
    df['rating_int'] = df['rating'].round().astype(int)
    
    print(f"\\n人工标注分布:")
    print(df['human_rating_int'].value_counts().sort_index())
    print(f"\\n预测结果分布:")
    print(df['rating_int'].value_counts().sort_index())
    
    return df

def calculate_main_metrics(y_true, y_pred):
    """计算主要指标"""
    print("\\n=== 主要指标 ===")
    
    # 1. QWK (Quadratic Weighted Kappa)
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    print(f"QWK (Quadratic Weighted Kappa): {qwk:.4f}")
    
    # QWK解释
    if qwk >= 0.8:
        qwk_interpretation = "优秀"
    elif qwk >= 0.6:
        qwk_interpretation = "良好"
    elif qwk >= 0.4:
        qwk_interpretation = "中等"
    elif qwk >= 0.2:
        qwk_interpretation = "一般"
    else:
        qwk_interpretation = "较差"
    print(f"QWK评级: {qwk_interpretation}")
    
    # 2. MAE (Mean Absolute Error)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"MAE (Mean Absolute Error): {mae:.4f}")
    
    # 3. Weighted F1 和 Macro F1
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    
    return {
        'qwk': qwk,
        'qwk_interpretation': qwk_interpretation,
        'mae': mae,
        'weighted_f1': weighted_f1,
        'macro_f1': macro_f1
    }

def calculate_auxiliary_metrics(y_true, y_pred):
    """计算辅助指标"""
    print("\\n=== 辅助指标 ===")
    
    # 1. Recall@HighRisk (对真实=4/5的召回率)
    high_risk_true = (y_true >= 4)
    high_risk_pred = (y_pred >= 4)
    
    if np.sum(high_risk_true) > 0:
        high_risk_recall = np.sum(high_risk_true & high_risk_pred) / np.sum(high_risk_true)
        print(f"Recall@HighRisk (真实4/5分的召回率): {high_risk_recall:.4f}")
        
        # 高风险精确率
        if np.sum(high_risk_pred) > 0:
            high_risk_precision = np.sum(high_risk_true & high_risk_pred) / np.sum(high_risk_pred)
            print(f"Precision@HighRisk (预测4/5分的精确率): {high_risk_precision:.4f}")
        else:
            high_risk_precision = 0.0
            print(f"Precision@HighRisk: 无预测为高风险的样本")
    else:
        high_risk_recall = 0.0
        high_risk_precision = 0.0
        print("无真实高风险样本")
    
    # 2. FPR@LowRisk (真实=1/2的误判率为高风险)
    low_risk_true = (y_true <= 2)
    low_risk_pred = (y_pred <= 2)
    false_positive_high = low_risk_true & (y_pred >= 4)
    
    if np.sum(low_risk_true) > 0:
        fpr_low_risk = np.sum(false_positive_high) / np.sum(low_risk_true)
        print(f"FPR@LowRisk (真实1/2分被误判为4/5分的比例): {fpr_low_risk:.4f}")
        
        # 低风险召回率和精确率
        if np.sum(low_risk_pred) > 0:
            low_risk_recall = np.sum(low_risk_true & low_risk_pred) / np.sum(low_risk_true)
            low_risk_precision = np.sum(low_risk_true & low_risk_pred) / np.sum(low_risk_pred)
            print(f"Recall@LowRisk (真实1/2分的召回率): {low_risk_recall:.4f}")
            print(f"Precision@LowRisk (预测1/2分的精确率): {low_risk_precision:.4f}")
        else:
            low_risk_recall = 0.0
            low_risk_precision = 0.0
            print(f"Recall@LowRisk: 无预测为低风险的样本")
    else:
        fpr_low_risk = 0.0
        low_risk_recall = 0.0
        low_risk_precision = 0.0
        print("无真实低风险样本")
    
    # 3. 高危和低危的Macro F1和Weighted F1
    # 创建二分类标签用于计算F1
    # 高危二分类 (4-5 vs 其他)
    y_true_high_binary = (y_true >= 4).astype(int)
    y_pred_high_binary = (y_pred >= 4).astype(int)
    
    # 低危二分类 (1-2 vs 其他)  
    y_true_low_binary = (y_true <= 2).astype(int)
    y_pred_low_binary = (y_pred <= 2).astype(int)
    
    # 计算高危的Macro F1和Weighted F1
    try:
        high_risk_macro_f1 = f1_score(y_true_high_binary, y_pred_high_binary, average='macro')
        high_risk_weighted_f1 = f1_score(y_true_high_binary, y_pred_high_binary, average='weighted')
        print(f"Macro F1@HighRisk (高危宏平均F1): {high_risk_macro_f1:.4f}")
        print(f"Weighted F1@HighRisk (高危加权F1): {high_risk_weighted_f1:.4f}")
    except Exception as e:
        high_risk_macro_f1 = 0.0
        high_risk_weighted_f1 = 0.0
        print(f"高危Macro/Weighted F1计算失败: {str(e)}")
    
    # 计算低危的Macro F1和Weighted F1
    try:
        low_risk_macro_f1 = f1_score(y_true_low_binary, y_pred_low_binary, average='macro')
        low_risk_weighted_f1 = f1_score(y_true_low_binary, y_pred_low_binary, average='weighted')
        print(f"Macro F1@LowRisk (低危宏平均F1): {low_risk_macro_f1:.4f}")
        print(f"Weighted F1@LowRisk (低危加权F1): {low_risk_weighted_f1:.4f}")
    except Exception as e:
        low_risk_macro_f1 = 0.0
        low_risk_weighted_f1 = 0.0
        print(f"低危Macro/Weighted F1计算失败: {str(e)}")
    
    # 5. Cost-aware Error (成本加权误差)
    # 将低风险误判为高风险的成本设为最高
    cost_matrix = np.array([
        [0, 1, 2, 4, 8],    # 真实=1, 预测为1,2,3,4,5的成本
        [1, 0, 1, 3, 6],    # 真实=2
        [2, 1, 0, 2, 4],    # 真实=3  
        [1, 2, 1, 0, 1],    # 真实=4
        [2, 3, 2, 1, 0]     # 真实=5
    ])
    
    total_cost = 0
    for i, true_val in enumerate(y_true):
        pred_val = y_pred[i]
        cost = cost_matrix[true_val-1, pred_val-1]
        total_cost += cost
    
    avg_cost = total_cost / len(y_true)
    print(f"Cost-aware Error (平均成本): {avg_cost:.4f}")
    
    return {
        'high_risk_recall': high_risk_recall,
        'high_risk_precision': high_risk_precision,
        'high_risk_macro_f1': high_risk_macro_f1,
        'high_risk_weighted_f1': high_risk_weighted_f1,
        'low_risk_recall': low_risk_recall,
        'low_risk_precision': low_risk_precision,
        'low_risk_macro_f1': low_risk_macro_f1,
        'low_risk_weighted_f1': low_risk_weighted_f1,
        'fpr_low_risk': fpr_low_risk,
        'avg_cost': avg_cost
    }

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """绘制混淆矩阵并分析错位方向"""
    print("\\n=== 混淆矩阵分析 ===")
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=[1,2,3,4,5])
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绝对数量混淆矩阵
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[1,2,3,4,5], yticklabels=[1,2,3,4,5], ax=ax1)
    ax1.set_title('混淆矩阵 (绝对数量)')
    ax1.set_xlabel('预测评分')
    ax1.set_ylabel('真实评分')
    
    # 比例混淆矩阵
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=[1,2,3,4,5], yticklabels=[1,2,3,4,5], ax=ax2)
    ax2.set_title('混淆矩阵 (行归一化比例)')
    ax2.set_xlabel('预测评分')
    ax2.set_ylabel('真实评分')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # 分析错位方向
    print("\\n混淆矩阵 (绝对数量):")
    print("真实\\\\预测", end="")
    for j in range(5):
        print(f"{j+1:6d}", end="")
    print()
    
    for i in range(5):
        print(f"{i+1:8d}", end="")
        for j in range(5):
            print(f"{cm[i][j]:6d}", end="")
        print()
    
    # 错位方向分析
    print("\\n=== 错位方向分析 ===")
    
    # 高估（预测高于真实）
    overestimate = 0
    underestimate = 0
    correct = 0
    
    for i in range(5):
        for j in range(5):
            if i < j:  # 预测高于真实
                overestimate += cm[i][j]
            elif i > j:  # 预测低于真实  
                underestimate += cm[i][j]
            else:  # 预测正确
                correct += cm[i][j]
    
    total = overestimate + underestimate + correct
    print(f"预测正确: {correct} ({correct/total*100:.1f}%)")
    print(f"高估 (预测>真实): {overestimate} ({overestimate/total*100:.1f}%)")
    print(f"低估 (预测<真实): {underestimate} ({underestimate/total*100:.1f}%)")
    
    # 严重错位分析（差距>=2）
    severe_errors = 0
    for i in range(5):
        for j in range(5):
            if abs(i - j) >= 2:
                severe_errors += cm[i][j]
    
    print(f"严重错位 (|预测-真实|>=2): {severe_errors} ({severe_errors/total*100:.1f}%)")
    
    return cm

def analyze_by_risk_level(df):
    """按风险等级详细分析"""
    print("\\n=== 按风险等级详细分析 ===")
    
    risk_levels = {
        1: "低风险", 2: "较低风险", 3: "中等风险", 
        4: "较高风险", 5: "高风险"
    }
    
    for level in [1, 2, 3, 4, 5]:
        subset = df[df['human_rating_int'] == level]
        if len(subset) == 0:
            continue
            
        print(f"\\n{risk_levels[level]} (真实评分={level}) - {len(subset)}个样本:")
        
        # 预测分布
        pred_dist = subset['rating_int'].value_counts().sort_index()
        for pred_level, count in pred_dist.items():
            percentage = count / len(subset) * 100
            print(f"  预测为{pred_level}: {count}个 ({percentage:.1f}%)")
        
        # 准确率
        accuracy = (subset['human_rating_int'] == subset['rating_int']).mean()
        print(f"  准确率: {accuracy:.3f}")
        
        # 平均预测误差
        mae = np.mean(np.abs(subset['human_rating_int'] - subset['rating_int']))
        print(f"  平均绝对误差: {mae:.3f}")

def plot_distribution_comparison(df, save_path=None):
    """绘制分布对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 整体分布对比
    ax1 = axes[0, 0]
    x = np.arange(1, 6)
    true_dist = df['human_rating_int'].value_counts().sort_index()
    pred_dist = df['rating_int'].value_counts().sort_index()
    
    width = 0.35
    ax1.bar(x - width/2, [true_dist.get(i, 0) for i in range(1, 6)], 
            width, label='人工标注', alpha=0.8)
    ax1.bar(x + width/2, [pred_dist.get(i, 0) for i in range(1, 6)], 
            width, label='预测结果', alpha=0.8)
    ax1.set_xlabel('评分')
    ax1.set_ylabel('数量')
    ax1.set_title('评分分布对比')
    ax1.legend()
    ax1.set_xticks(x)
    
    # 2. 散点图
    ax2 = axes[0, 1]
    ax2.scatter(df['human_rating_int'], df['rating_int'], alpha=0.6)
    ax2.plot([1, 5], [1, 5], 'r--', label='完美预测线')
    ax2.set_xlabel('人工标注')
    ax2.set_ylabel('预测结果')
    ax2.set_title('预测 vs 真实 散点图')
    ax2.legend()
    ax2.set_xlim(0.5, 5.5)
    ax2.set_ylim(0.5, 5.5)
    
    # 3. 误差分布
    ax3 = axes[1, 0]
    errors = df['rating_int'] - df['human_rating_int']
    ax3.hist(errors, bins=np.arange(-4.5, 5.5, 1), alpha=0.7, edgecolor='black')
    ax3.set_xlabel('预测误差 (预测-真实)')
    ax3.set_ylabel('频次')
    ax3.set_title('预测误差分布')
    ax3.axvline(x=0, color='red', linestyle='--', label='零误差')
    ax3.legend()
    
    # 4. 按真实评分的预测准确性
    ax4 = axes[1, 1]
    accuracies = []
    for level in range(1, 6):
        subset = df[df['human_rating_int'] == level]
        if len(subset) > 0:
            acc = (subset['human_rating_int'] == subset['rating_int']).mean()
            accuracies.append(acc)
        else:
            accuracies.append(0)
    
    ax4.bar(range(1, 6), accuracies, alpha=0.8)
    ax4.set_xlabel('真实评分')
    ax4.set_ylabel('准确率')
    ax4.set_title('各评分等级预测准确率')
    ax4.set_xticks(range(1, 6))
    ax4.set_ylim(0, 1)
    
    # 添加数值标签
    for i, acc in enumerate(accuracies):
        ax4.text(i+1, acc + 0.01, f'{acc:.3f}', ha='center')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def generate_report(main_metrics, aux_metrics, total_samples):
    """生成评估报告"""
    print("\\n" + "="*60)
    print("反欺诈模型评估报告")
    print("="*60)
    
    print(f"\\n数据概况:")
    print(f"- 总样本数: {total_samples:,}")
    
    print(f"\\n主要指标:")
    print(f"- QWK (Quadratic Weighted Kappa): {main_metrics['qwk']:.4f} ({main_metrics['qwk_interpretation']})")
    print(f"- MAE (Mean Absolute Error): {main_metrics['mae']:.4f}")
    print(f"- Weighted F1: {main_metrics['weighted_f1']:.4f}")
    print(f"- Macro F1: {main_metrics['macro_f1']:.4f}")
    
    print(f"\\n辅助指标:")
    print(f"- 高风险召回率 (Recall@HighRisk): {aux_metrics['high_risk_recall']:.4f}")
    print(f"- 高风险精确率 (Precision@HighRisk): {aux_metrics['high_risk_precision']:.4f}")
    print(f"- 高风险宏平均F1 (Macro F1@HighRisk): {aux_metrics['high_risk_macro_f1']:.4f}")
    print(f"- 高风险加权F1 (Weighted F1@HighRisk): {aux_metrics['high_risk_weighted_f1']:.4f}")
    print(f"- 低风险召回率 (Recall@LowRisk): {aux_metrics['low_risk_recall']:.4f}")
    print(f"- 低风险精确率 (Precision@LowRisk): {aux_metrics['low_risk_precision']:.4f}")
    print(f"- 低风险宏平均F1 (Macro F1@LowRisk): {aux_metrics['low_risk_macro_f1']:.4f}")
    print(f"- 低风险加权F1 (Weighted F1@LowRisk): {aux_metrics['low_risk_weighted_f1']:.4f}")
    print(f"- 低风险误判率 (FPR@LowRisk): {aux_metrics['fpr_low_risk']:.4f}")
    print(f"- 成本加权误差 (Cost-aware Error): {aux_metrics['avg_cost']:.4f}")
    
    print(f"\\n模型评估结论:")
    
    # QWK评估
    if main_metrics['qwk'] >= 0.8:
        print("✅ QWK表现优秀，模型预测与人工标注高度一致")
    elif main_metrics['qwk'] >= 0.6:
        print("🟡 QWK表现良好，模型预测基本可靠")
    else:
        print("❌ QWK表现不佳，模型需要进一步优化")
    
    # MAE评估
    if main_metrics['mae'] <= 0.5:
        print("✅ MAE表现优秀，平均误差很小")
    elif main_metrics['mae'] <= 1.0:
        print("🟡 MAE表现中等，存在一定误差")
    else:
        print("❌ MAE较大，预测误差明显")
    
    # 高风险检测评估
    if aux_metrics['high_risk_recall'] >= 0.9:
        print("✅ 高风险检测能力优秀，漏检率低")
    elif aux_metrics['high_risk_recall'] >= 0.8:
        print("🟡 高风险检测能力良好")
    else:
        print("❌ 高风险检测能力不足，可能存在漏检问题")
    
    # 误判率评估
    if aux_metrics['fpr_low_risk'] <= 0.05:
        print("✅ 低风险误判率很低，误报控制良好")
    elif aux_metrics['fpr_low_risk'] <= 0.1:
        print("🟡 低风险误判率较低")
    else:
        print("❌ 低风险误判率偏高，可能存在过度敏感问题")
    
    # Macro F1指标评估
    if aux_metrics['high_risk_macro_f1'] >= 0.8:
        print("✅ 高风险宏平均F1分数优秀，高危检测综合性能良好")
    elif aux_metrics['high_risk_macro_f1'] >= 0.6:
        print("🟡 高风险宏平均F1分数良好")
    else:
        print("❌ 高风险宏平均F1分数偏低，需要在召回率和精确率间平衡")
    
    if aux_metrics['low_risk_macro_f1'] >= 0.8:
        print("✅ 低风险宏平均F1分数优秀，低危识别综合性能良好")
    elif aux_metrics['low_risk_macro_f1'] >= 0.6:
        print("🟡 低风险宏平均F1分数良好")
    else:
        print("❌ 低风险宏平均F1分数偏低，需要优化低危样本识别能力")

def main():
    """主函数"""
    # 文件路径
    input_file = "result.csv"
    
    print("反欺诈模型评估脚本")
    print("="*50)
    
    # 1. 数据加载和预处理
    df = load_and_preprocess_data(input_file)
    
    if len(df) == 0:
        print("错误: 没有有效数据进行评估")
        return
    
    # 提取评分数据
    y_true = df['human_rating_int'].values
    y_pred = df['rating_int'].values
    
    # 2. 计算主要指标
    main_metrics = calculate_main_metrics(y_true, y_pred)
    
    # 3. 计算辅助指标  
    aux_metrics = calculate_auxiliary_metrics(y_true, y_pred)
    
    # 4. 绘制混淆矩阵
    cm = plot_confusion_matrix(y_true, y_pred, "confusion_matrix.png")
    
    # 5. 按风险等级分析
    analyze_by_risk_level(df)
    
    # 6. 绘制分布对比图
    plot_distribution_comparison(df, "distribution_comparison.png")
    
    # 7. 生成完整报告
    generate_report(main_metrics, aux_metrics, len(df))
    
    # 8. 保存详细结果
    results = {
        **main_metrics,
        **aux_metrics,
        'total_samples': len(df)
    }
    
    # 保存为JSON
    import json
    with open('evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\\n评估完成！")
    print(f"- 详细结果已保存到: evaluation_results.json")
    print(f"- 混淆矩阵图已保存到: confusion_matrix.png") 
    print(f"- 分布对比图已保存到: distribution_comparison.png")

if __name__ == "__main__":
    main()