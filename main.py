import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import matplotlib.colors as mcolors
import warnings
from glob import glob
import os
import time
import json
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def create_figure_folder(folder_name='code_2_analysis_figure'):
    """创建保存图表的文件夹"""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"创建文件夹: {folder_name}")
    return folder_name

def demographic_analysis(df, output_folder):
    """人口统计学特征分析 - 多个独立图表"""
    print("开始人口统计学特征分析...")

    sample_size = 20000
    
    # 1-1 年龄分布直方图
    plt.figure(figsize=(12, 6))
    sns.histplot(df['age'], bins=20, kde=True)
    plt.title('用户年龄分布(直方图)', fontsize=14)
    plt.xlabel('年龄')
    plt.ylabel('用户数量')
    plt.tight_layout()
    plt.savefig(f'{output_folder}/1-1_用户年龄分布直方图.png', dpi=300)
    plt.close()
    
    # 1-2 收入分布 - 柱状图
    plt.figure(figsize=(12, 8))
    income_counts = df['income_group'].value_counts().sort_index()
    ax = sns.barplot(x=income_counts.index, y=income_counts.values, palette='magma')
    
    y_min = income_counts.min() * 0.95  
    y_max = income_counts.max() * 1.02  
    plt.ylim(y_min, y_max)
    
    

    for i, v in enumerate(income_counts.values):
        ax.text(i, v + y_max*0.01, f'{v:,}', ha='center', fontsize=10)
    
    plt.title('用户收入分布', fontsize=14)
    plt.xlabel('收入组')
    plt.ylabel('用户数量')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_folder}/1-2_用户收入分布.png', dpi=300)
    plt.close()
    
    # 1-3 地理分布 - 所有省份条形图
    plt.figure(figsize=(15, 12))  
    province_counts = df['province'].value_counts()
    sns.barplot(x=province_counts.values, y=province_counts.index, palette='Blues_d')
    plt.title('用户地理分布(所有地区)', fontsize=14)
    plt.xlabel('用户数量')
    plt.ylabel('省份')
    plt.tight_layout()
    plt.savefig(f'{output_folder}/1-3_用户地理分布所有地区.png', dpi=300)
    plt.close()
    
    # 1-4 信用评分分布KDE
    plt.figure(figsize=(12, 6))
    sns.histplot(df['credit_score'], bins=20, kde=True)
    plt.title('用户信用评分分布(带密度曲线)', fontsize=14)
    plt.xlabel('信用评分')
    plt.ylabel('用户数量')
    plt.tight_layout()
    plt.savefig(f'{output_folder}/1-4_用户信用评分分布KDE.png', dpi=300)
    plt.close()
    
    # 1-5 年龄-收入热力图
    plt.figure(figsize=(12, 8))
    df_sample = df[['age', 'income']].sample(min(sample_size, len(df)), random_state=42)
    print(f"热力图分析使用抽样数据：{len(df_sample)}条")
    
    age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    age_labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91+']
    df_sample['age_bin'] = pd.cut(df_sample['age'], bins=age_bins, labels=age_labels)
    
    income_bins = [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000]
    income_labels = ['0-10万', '10-20万', '20-30万', '30-40万', '40-50万', '50-60万', '60-70万', '70-80万', '80-90万', '90-100万']
    df_sample['income_bin'] = pd.cut(df_sample['income'], bins=income_bins, labels=income_labels)
    
    # 计算每个组合的人数
    heatmap_data = pd.crosstab(df_sample['age_bin'], df_sample['income_bin'])
    
    # 绘制热力图
    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='Reds', cbar_kws={'label': '人数'})
    plt.title(f'年龄-收入分布热力图 - 抽样{len(df_sample)}条数据', fontsize=14)
    plt.xlabel('收入水平')
    plt.ylabel('年龄段')
    plt.tight_layout()
    plt.savefig(f'{output_folder}/1-5_年龄收入热力图.png', dpi=300)
    plt.close()

    # 释放内存
    del df_sample
    
    # 1-6 信用评分与收入关系
    plt.figure(figsize=(10, 6))
    # 重新随机抽样，只选择需要的列
    df_sample = df[['credit_score', 'income', 'age_group']].sample(min(sample_size, len(df)), random_state=42)
    sns.scatterplot(data=df_sample, x='credit_score', y='income', hue='age_group', alpha=0.6)
    plt.title(f'信用评分与收入关系(按年龄段) - 抽样{len(df_sample)}条数据', fontsize=14)
    plt.xlabel('信用评分')
    plt.ylabel('收入')
    plt.tight_layout()
    plt.savefig(f'{output_folder}/1-6_信用评分收入年龄关系.png', dpi=300)
    plt.close()
    
    # 释放内存
    del df_sample
    
    # 1-7 地域分布与收入水平
    if 'province' in df.columns:
        # 按省份聚合而不是使用全量数据
        province_stats = df.groupby('province').agg({
            'income': 'mean',
            'id': 'count'
        }).reset_index()
        province_stats = province_stats.sort_values('id', ascending=False).head(10)
        
        # 双坐标轴图表
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        # 左侧Y轴 - 用户数量
        bars = ax1.bar(province_stats['province'], province_stats['id'], color='skyblue')
        ax1.set_xlabel('省份')
        ax1.set_ylabel('用户数量', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # 右侧Y轴 - 平均收入
        ax2 = ax1.twinx()
        line = ax2.plot(province_stats['province'], province_stats['income'], 'ro-')
        ax2.set_ylabel('平均收入', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        plt.title('Top 10省份的用户数量与平均收入', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{output_folder}/1-7_省份用户数量与平均收入.png', dpi=300)
        plt.close()
        
        # 释放内存
        del province_stats
    
    print("人口统计学特征分析完成")
    return "人口统计学分析完成"


def shopping_behavior_analysis(df, output_folder):
    """购物行为分析 - 独立图表"""
    print("开始购物行为分析...")
    
    # 2-1 不同年龄组的平均消费
    plt.figure(figsize=(12, 8))
    age_spend = df.groupby('age_group')['total_spent'].mean().sort_index()
    ax = sns.barplot(x=age_spend.index, y=age_spend.values, palette='YlOrRd')

    # 控制y轴范围
    y_min = age_spend.min() * 0.95  # 从0开始
    y_max = age_spend.max() * 1.02  # 最大值上方留出10%空间
    plt.ylim(y_min, y_max)

    # 在柱状图上显示具体数值
    for i, v in enumerate(age_spend.values):
        ax.text(i, v + (y_max * 0.01), f'{v:.2f}', ha='center', fontsize=10)

    # 添加网格线增强可读性
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.title('不同年龄组的平均消费', fontsize=14)
    plt.xlabel('年龄组')
    plt.ylabel('平均消费金额')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_folder}/2-1_不同年龄组的平均消费.png', dpi=300)
    plt.close()
    
    # 2-2 不同性别的消费习惯比较
    plt.figure(figsize=(12, 8))
    gender_metrics = df.groupby('gender').agg({
        'total_spent': 'mean',
        'order_count': 'mean',
        'avg_price': 'mean'
    }).reset_index()
    
    # 转换为长格式用于seaborn绘图
    gender_metrics_long = pd.melt(
        gender_metrics, 
        id_vars='gender', 
        value_vars=['total_spent', 'order_count', 'avg_price'],
        var_name='指标', 
        value_name='值'
    )
    
    # 标准化值以便比较
    for metric in ['total_spent', 'order_count', 'avg_price']:
        max_val = gender_metrics_long[gender_metrics_long['指标'] == metric]['值'].max()
        gender_metrics_long.loc[gender_metrics_long['指标'] == metric, '值'] = \
            gender_metrics_long.loc[gender_metrics_long['指标'] == metric, '值'] / max_val

    ax = sns.barplot(x='gender', y='值', hue='指标', data=gender_metrics_long, palette='Set2')

    # 控制y轴范围
    y_min = 0.95  # 标准化值从0开始
    y_max = 1.03  # 最大标准化值为1，上方留出10%空间
    plt.ylim(y_min, y_max)

    # 在柱状图上显示具体数值
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 0.02, 
                f'{height:.2f}', ha='center', fontsize=9)

    # 添加网格线增强可读性
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.title('不同性别的消费习惯比较（标准化）', fontsize=14)
    plt.xlabel('性别')
    plt.ylabel('标准化值')
    plt.legend(title='指标')
    plt.tight_layout()
    plt.savefig(f'{output_folder}/2-2_不同性别的消费习惯比较.png', dpi=300)
    plt.close()
    
    # 2-3 活跃天数与消费金额的散点图
    plt.figure(figsize=(12, 8))
    # 限制数据范围以便更好地可视化
    plt.scatter(
        df['active_days'].clip(upper=df['active_days'].quantile(0.95)), 
        df['total_spent'].clip(upper=df['total_spent'].quantile(0.95)), 
        alpha=0.5, c=df['age'], cmap='viridis'
    )
    plt.colorbar(label='年龄')
    plt.title('用户活跃天数与总消费关系(按年龄着色)', fontsize=14)
    plt.xlabel('活跃天数')
    plt.ylabel('总消费金额')
    plt.tight_layout()
    plt.savefig(f'{output_folder}/2-3_用户活跃天数与总消费关系.png', dpi=300)
    plt.close()
    
    # 2-4 消费活动时段分布 - 按工作日/周末分组
    plt.figure(figsize=(12, 8))
    # 创建时段分布数据
    weekday_hours = df[df['is_weekend'] == 0]['activity_hour'].value_counts().sort_index()
    weekend_hours = df[df['is_weekend'] == 1]['activity_hour'].value_counts().sort_index()
    
    # 确保两个Series有相同的索引
    all_hours = range(24)
    weekday_data = pd.Series([weekday_hours.get(hour, 0) for hour in all_hours], index=all_hours)
    weekend_data = pd.Series([weekend_hours.get(hour, 0) for hour in all_hours], index=all_hours)
    
    # 绘制双线图
    plt.plot(weekday_data.index, weekday_data.values, 'b-', label='工作日')
    plt.plot(weekend_data.index, weekend_data.values, 'r-', label='周末')
    plt.title('工作日vs周末消费活动时段分布', fontsize=14)
    plt.xlabel('小时')
    plt.ylabel('活动次数')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{output_folder}/2-4_工作日vs周末消费活动时段分布.png', dpi=300)
    plt.close()
    
    return "购物行为分析完成"


def analyze_user_profiles(df, output_folder=None):
    """执行用户画像分析，调用各个分析函数"""
    print("开始用户画像分析...")
    
    # 创建保存图表的目录（如果不存在）
    if output_folder is None:
        output_folder = create_figure_folder('code_2_analysis_figure')
    else:
        output_folder = create_figure_folder(output_folder)
    
    # 1. 执行人口统计学特征分析
    demographic_analysis(df, output_folder)
    
    # 2. 执行购物行为分析
    shopping_behavior_analysis(df, output_folder)
    

    
    return f"用户画像分析完成，分析图表已保存至 {output_folder} 文件夹"

def read_processed_files(folder_path="processed_data", columns=None, sample_size=None):
    """读取processed_data文件夹中的7个parquet文件并合并为一个DataFrame"""
    print("开始读取processed_data文件夹中的parquet文件...")
    start_time = time.time()
    
    # 定义需要的列，如果未指定则使用所有列
    if columns is None:
        # 这里列出所有可能需要的列
        columns = [
            'id', 'user_name', 'chinese_name', 'email', 'age', 'income', 'gender', 
            'country', 'chinese_address', 'credit_score', 'phone_number',    
            'registration_date', 'timestamp', 'is_active', 'quantity', 'purchase_data', 
            'category', 'avg_price', 'order_count', 'unique_categories', 'total_spent', 
            'active_days', 'reg_year', 'reg_month', 'activity_hour', 'activity_dayofweek', 
            'is_weekend', 'age_group', 'income_group', 'province'
        ]
    
    # 获取所有parquet文件路径
    parquet_files = glob(os.path.join(folder_path, "*.parquet"))
    print(f"找到 {len(parquet_files)} 个parquet文件")
    
    # 限制为前7个文件
    # parquet_files = parquet_files[:1]
    dfs = []
    
    # 逐个读取文件
    for i, file_path in enumerate(parquet_files):
        print(f"正在读取文件 {i+1}/{len(parquet_files)}: {os.path.basename(file_path)}")
        file_start_time = time.time()
        
        try:
            # 读取指定列的数据
            df = pd.read_parquet(file_path, columns=columns)
            
            # 如果指定了样本大小，对读取的数据进行采样
            if sample_size and len(df) > sample_size:
                df = df.sample(sample_size, random_state=42)
                
            dfs.append(df)
            
            file_end_time = time.time()
            print(f"  文件读取完成，耗时: {file_end_time - file_start_time:.2f}秒，文件大小: {df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")
        
        except Exception as e:
            print(f"  读取文件时出错: {str(e)}")
            # 尝试不指定列名读取，然后选择列和采样
            try:
                print("  尝试以默认方式读取文件...")
                df = pd.read_parquet(file_path)
                
                # 只保留需要的列
                available_cols = [col for col in columns if col in df.columns]
                df = df[available_cols]
                
                # 采样
                if sample_size and len(df) > sample_size:
                    df = df.sample(sample_size, random_state=42)
                    
                dfs.append(df)
                
                file_end_time = time.time()
                print(f"  文件读取完成，耗时: {file_end_time - file_start_time:.2f}秒，文件大小: {df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")
            except Exception as e2:
                print(f"  无法读取文件: {str(e2)}")
                continue
    
    if not dfs:
        raise ValueError("没有成功读取任何数据")
    
    # 合并所有数据框
    print("合并所有数据...")
    result = pd.concat(dfs, ignore_index=True)
    
    # 优化内存使用
    # 转换数值型列为更小的数据类型
    for col in result.select_dtypes(include=['int']).columns:
        result[col] = pd.to_numeric(result[col], downcast='integer')
    for col in result.select_dtypes(include=['float']).columns:
        result[col] = pd.to_numeric(result[col], downcast='float')
    
    # 转换分类数据
    for col in ['gender', 'age_group', 'income_group', 'province']:
        if col in result.columns:
            result[col] = result[col].astype('category')
    
    end_time = time.time()
    print(f"所有文件读取并合并完成，总耗时: {end_time - start_time:.2f}秒")
    print(f"总数据大小: {result.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")
    
    return result

# 主函数
def main():
    # 读取并合并processed_data文件夹中的parquet文件
    df = read_processed_files("processed_data")
    
    print("\n开始处理数据...")
    # 执行用户画像分析，指定输出文件夹
    result = analyze_user_profiles(df, 'code_2_whole_analysis_figure')
    print(result)

if __name__ == "__main__":
    main()
