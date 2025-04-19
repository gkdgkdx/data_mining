import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob
import time
import json
import gc
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def create_figure_folder(folder_name):
    """创建保存图表的文件夹"""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"创建文件夹: {folder_name}")
    return folder_name

def read_data_for_chart(folder_path="new_processed_data_30G", columns=None, sample_size=None):
    """读取指定列的数据，针对特定图表优化内存使用"""
    # 获取所有parquet文件路径
    parquet_files = glob(os.path.join(folder_path, "*.parquet"))
    print(f"找到 {len(parquet_files)} 个parquet文件")
    
    all_data = []
    
    # 逐个读取文件
    for file_path in tqdm(parquet_files, desc="读取文件进度"):
        try:
            # 只读取需要的列
            df = pd.read_parquet(file_path, columns=columns)
            
            # 如果指定了样本大小，对读取的数据进行采样
            if sample_size and len(df) > sample_size:
                df = df.sample(sample_size // len(parquet_files), random_state=42)
            
            # 优化数据类型
            for col in df.select_dtypes(include=['int']).columns:
                df[col] = pd.to_numeric(df[col], downcast='integer')
            for col in df.select_dtypes(include=['float']).columns:
                df[col] = pd.to_numeric(df[col], downcast='float')
            
            # 转换分类列 - 根据新的数据结构调整
            categorical_columns = ['gender', 'country', 'province', 'age_group', 'income_group']
            for col in categorical_columns:
                if col in df.columns:
                    df[col] = df[col].astype('category')
            
            all_data.append(df)
        except Exception as e:
            print(f"读取文件 {file_path} 失败: {e}")
    
    # 合并数据
    if not all_data:
        raise ValueError("没有成功读取任何数据")
    
    result = pd.concat(all_data, ignore_index=True)
    
    # 再次采样以确保总样本量不超过指定值
    if sample_size and len(result) > sample_size:
        result = result.sample(sample_size, random_state=42)
    
    return result

def plot_age_distribution(output_folder, sample_size=None):
    """1-1 年龄分布直方图"""
    print("生成年龄分布直方图...")
    
    # 只读取需要的列
    df = read_data_for_chart(columns=['age'], sample_size=sample_size)
    
    plt.figure(figsize=(12, 6))
    ax = sns.histplot(df['age'], bins=20, kde=True)
    
    # 获取频数值
    heights = [p.get_height() for p in ax.patches]
    y_min = min(heights) * 0.95
    y_max = max(heights) * 1.02
    plt.ylim(y_min, y_max)
    
    plt.title('用户年龄分布(直方图)', fontsize=14)
    plt.xlabel('年龄')
    plt.ylabel('用户数量')
    plt.tight_layout()
    plt.savefig(f'{output_folder}/1-1_用户年龄分布直方图.png', dpi=300)
    plt.close()
    
    # 清理内存
    del df
    gc.collect()
    print("年龄分布直方图生成完成")

def plot_income_distribution(output_folder, sample_size=None):
    """1-2 收入分布柱状图"""
    print("生成收入分布柱状图...")
    
    # 只读取需要的列
    df = read_data_for_chart(columns=['income_group'], sample_size=sample_size)
    
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
    
    # 清理内存
    del df
    gc.collect()
    print("收入分布柱状图生成完成")

def plot_geographic_distribution(output_folder, sample_size=None):
    """1-3 地理分布条形图"""
    print("生成地理分布条形图...")
    
    # 只读取需要的列
    df = read_data_for_chart(columns=['province'], sample_size=sample_size)
    
    # 计算各省份用户数量并排序
    province_counts = df['province'].value_counts().sort_values(ascending=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图表，设置合适的大小
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 创建水平条形图
    bars = ax.barh(range(len(province_counts)), province_counts.values)
    
    # 设置y轴刻度标签
    ax.set_yticks(range(len(province_counts)))
    ax.set_yticklabels(province_counts.index, fontsize=10)
    
    # 设置x轴范围，突出显示差异
    x_min = min(province_counts) * 0.99
    x_max = max(province_counts) * 1.01
    ax.set_xlim(x_min, x_max)
    
    # 设置x轴标签
    ax.set_xlabel('用户数量', fontsize=12)
    
    # 在条形上添加数值标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                f'{int(width):,}',
                ha='left', va='center', fontsize=9)
    
    # 添加标题
    plt.title('用户地理分布(所有地区)', fontsize=14, pad=20)
    
    # 添加网格线
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    plt.savefig(f'{output_folder}/1-3_用户地理分布所有地区.png', dpi=300)
    plt.close()
    
    # 输出数据统计
    print("\n各省份用户分布情况：")
    for province, count in province_counts.items():
        print(f"{province}: {count:,}人")
    
    # 清理内存
    del df
    gc.collect()
    print("\n地理分布条形图生成完成")

def plot_country_distribution(output_folder, sample_size=None):
    """1-4 国家分布图"""
    print("生成国家分布图...")
    
    # 只读取需要的列
    df = read_data_for_chart(columns=['country'], sample_size=sample_size)
    
    plt.figure(figsize=(14, 10))  
    country_counts = df['country'].value_counts().head(15)  # 取前15个国家
    ax = sns.barplot(x=country_counts.values, y=country_counts.index, palette='viridis')
    
    # 设置x轴范围
    x_min = country_counts.min() * 0.95
    x_max = country_counts.max() * 1.02
    plt.xlim(x_min, x_max)
    
    # 在条形上添加数值标签
    for i, v in enumerate(country_counts.values):
        ax.text(v + x_max*0.01, i, f'{v:,}', va='center', fontsize=10)
    
    plt.title('用户国家分布(Top 15)', fontsize=14)
    plt.xlabel('用户数量')
    plt.ylabel('国家')
    plt.tight_layout()
    plt.savefig(f'{output_folder}/1-4_用户国家分布.png', dpi=300)
    plt.close()
    
    # 清理内存
    del df
    gc.collect()
    print("国家分布图生成完成")

def plot_age_income_heatmap(output_folder, sample_size=20000):
    """1-5 年龄-收入热力图"""
    print("生成年龄-收入热力图...")
    
    # 只读取需要的列 - 使用已经分组的数据
    df = read_data_for_chart(columns=['age_group', 'income_group'], sample_size=sample_size)
    
    plt.figure(figsize=(12, 8))
    
    # 直接使用已经分组的数据创建热力图
    heatmap_data = pd.crosstab(df['age_group'], df['income_group'])
    
    # 绘制热力图
    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='Reds', cbar_kws={'label': '人数'})
    plt.title(f'年龄-收入分布热力图 - 抽样{len(df)}条数据', fontsize=14)
    plt.xlabel('收入水平')
    plt.ylabel('年龄段')
    plt.tight_layout()
    plt.savefig(f'{output_folder}/1-5_年龄收入热力图.png', dpi=300)
    plt.close()
    
    # 清理内存
    del df
    gc.collect()
    print("年龄-收入热力图生成完成")

def plot_session_duration_by_age(output_folder, sample_size=None):
    """1-6 不同年龄组的平均会话时长"""
    print("生成不同年龄组的平均会话时长图...")
    
    # 只读取需要的列
    df = read_data_for_chart(columns=['age_group', 'login_avg_session_duration'], sample_size=sample_size)
    
    plt.figure(figsize=(12, 8))
    session_by_age = df.groupby('age_group')['login_avg_session_duration'].mean().sort_index()
    ax = sns.barplot(x=session_by_age.index, y=session_by_age.values, palette='YlGnBu')

    # 控制y轴范围
    y_min = session_by_age.min() * 0.95
    y_max = session_by_age.max() * 1.02
    plt.ylim(y_min, y_max)

    # 在柱状图上显示具体数值
    for i, v in enumerate(session_by_age.values):
        ax.text(i, v + (y_max * 0.01), f'{v:.2f}', ha='center', fontsize=10)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.title('不同年龄组的平均会话时长', fontsize=14)
    plt.xlabel('年龄组')
    plt.ylabel('平均会话时长(秒)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_folder}/1-6_不同年龄组的平均会话时长.png', dpi=300)
    plt.close()
    
    # 清理内存
    del df
    gc.collect()
    print("不同年龄组的平均会话时长图生成完成")


def plot_device_usage(output_folder, sample_size=None):
    """2-1 设备使用情况"""
    print("生成设备使用情况图...")
    
    # 只读取需要的列
    df = read_data_for_chart(columns=['login_devices'], sample_size=sample_size)
    
    # 处理设备数据，拆分设备字段
    all_devices = []
    for devices_str in df['login_devices']:
        if isinstance(devices_str, str) and devices_str:
            devices = devices_str.split('|')
            all_devices.extend(devices)
    
    # 计算每种设备的使用次数
    device_counts = pd.Series(all_devices).value_counts()
    
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x=device_counts.values, y=device_counts.index, palette='viridis')
    
    # 设置x轴范围
    x_min = device_counts.min() * 0.95
    x_max = device_counts.max() * 1.02
    plt.xlim(x_min, x_max)
    
    # 添加数值标签
    for i, v in enumerate(device_counts.values):
        ax.text(v + x_max*0.01, i, f'{v:,}', va='center', fontsize=10)
    
    plt.title('用户设备使用情况', fontsize=14)
    plt.xlabel('使用次数')
    plt.ylabel('设备类型')
    plt.tight_layout()
    plt.savefig(f'{output_folder}/2-1_设备使用情况.png', dpi=300)
    plt.close()
    
    # 清理内存
    del df, device_counts
    gc.collect()
    print("设备使用情况图生成完成")

def plot_purchase_categories(output_folder, sample_size=None):
    """2-2 购买类别分布"""
    print("生成购买类别分布图...")
    
    # 只读取需要的列
    df = read_data_for_chart(columns=['purchase_categories'], sample_size=sample_size)
    
    # 统计不同类别的出现次数
    category_counts = df['purchase_categories'].value_counts().head(15)  # 只取前15个类别
    
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x=category_counts.values, y=category_counts.index, palette='coolwarm')
    
    # 设置x轴范围
    x_min = category_counts.min() * 0.95
    x_max = category_counts.max() * 1.02
    plt.xlim(x_min, x_max)
    
    # 添加数值标签
    for i, v in enumerate(category_counts.values):
        ax.text(v + x_max*0.01, i, f'{v:,}', va='center', fontsize=10)
    
    plt.title('热门购买类别分布(Top 15)', fontsize=14)
    plt.xlabel('出现次数')
    plt.ylabel('商品类别')
    plt.tight_layout()
    plt.savefig(f'{output_folder}/2-2_购买类别分布.png', dpi=300)
    plt.close()
    
    # 清理内存
    del df
    gc.collect()
    print("购买类别分布图生成完成")

def plot_payment_method_distribution(output_folder, sample_size=None):
    """2-3 支付方式分布"""
    print("生成支付方式分布图...")
    
    # 只读取需要的列
    df = read_data_for_chart(columns=['purchase_payment_method'], sample_size=sample_size)
    
    plt.figure(figsize=(12, 8))
    payment_counts = df['purchase_payment_method'].value_counts()
    
    # 饼图展示支付方式分布
    plt.pie(payment_counts, labels=payment_counts.index, autopct='%1.1f%%', 
            startangle=90, shadow=True, explode=[0.05]*len(payment_counts))
    plt.title('用户支付方式分布', fontsize=14)
    plt.axis('equal')  # 使饼图为正圆形
    plt.tight_layout()
    plt.savefig(f'{output_folder}/2-3_支付方式分布.png', dpi=300)
    plt.close()
    
    # 清理内存
    del df
    gc.collect()
    print("支付方式分布图生成完成")

def plot_payment_status_by_gender(output_folder, sample_size=None):
    """2-4 不同性别的支付状态分布"""
    print("生成不同性别的支付状态分布图...")
    
    # 只读取需要的列
    df = read_data_for_chart(columns=['gender', 'purchase_payment_status'], sample_size=sample_size)
    
    # 创建支付状态与性别的交叉表
    status_gender = pd.crosstab(df['purchase_payment_status'], df['gender'])
    
    # 计算百分比
    status_gender_pct = status_gender.div(status_gender.sum(axis=0), axis=1) * 100
    
    plt.figure(figsize=(12, 8))
    status_gender_pct.plot(kind='bar', stacked=False)
    plt.title('不同性别的支付状态分布(百分比)', fontsize=14)
    plt.xlabel('支付状态')
    plt.ylabel('百分比(%)')
    plt.legend(title='性别')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{output_folder}/2-4_不同性别的支付状态分布.png', dpi=300)
    plt.close()
    
    # 清理内存
    del df
    gc.collect()
    print("不同性别的支付状态分布图生成完成")

def plot_account_age_vs_purchases(output_folder, sample_size=None):
    """2-5 账户年龄与购买关系"""
    print("生成账户年龄与购买关系图...")
    
    # 只读取需要的列
    df = read_data_for_chart(columns=['account_age_days', 'purchase_items_count'], sample_size=sample_size)
    
    # 创建账户年龄分组
    df['account_age_group'] = pd.cut(df['account_age_days'], 
                                     bins=[0, 30, 90, 180, 365, 730, 1095, np.inf],
                                     labels=['<1月', '1-3月', '3-6月', '6-12月', '1-2年', '2-3年', '>3年'])
    
    # 计算每个账户年龄组的平均购买数量
    age_purchase = df.groupby('account_age_group')['purchase_items_count'].mean().reset_index()
    
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='account_age_group', y='purchase_items_count', data=age_purchase, palette='YlOrRd')
    
    # 设置y轴范围
    y_min = age_purchase['purchase_items_count'].min() * 0.95
    y_max = age_purchase['purchase_items_count'].max() * 1.02
    plt.ylim(y_min, y_max)
    
    # 添加数值标签
    for i, v in enumerate(age_purchase['purchase_items_count']):
        ax.text(i, v + y_max*0.01, f'{v:.2f}', ha='center', fontsize=10)
    
    plt.title('账户年龄与平均购买数量关系', fontsize=14)
    plt.xlabel('账户年龄')
    plt.ylabel('平均购买数量')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{output_folder}/2-5_账户年龄与购买关系.png', dpi=300)
    plt.close()
    
    # 清理内存
    del df
    gc.collect()
    print("账户年龄与购买关系图生成完成")


def plot_login_locations(output_folder, sample_size=None):
    """2-6 登录位置分布"""
    print("生成登录位置分布图...")
    
    # 只读取需要的列
    df = read_data_for_chart(columns=['login_locations'], sample_size=sample_size)
    
    # 处理位置数据，拆分位置字段
    all_locations = []
    for locations_str in df['login_locations']:
        if isinstance(locations_str, str) and locations_str:
            locations = locations_str.split('|')
            all_locations.extend(locations)
    
    # 计算每个位置的出现次数
    location_counts = pd.Series(all_locations).value_counts()
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=location_counts.values, y=location_counts.index, palette='Set3')
    
    # 设置x轴范围
    x_min = location_counts.min() * 0.95
    x_max = location_counts.max() * 1.02
    plt.xlim(x_min, x_max)
    
    # 添加数值标签
    for i, v in enumerate(location_counts.values):
        ax.text(v + x_max*0.01, i, f'{v:,}', va='center', fontsize=10)
    
    plt.title('用户登录位置分布', fontsize=14)
    plt.xlabel('出现次数')
    plt.ylabel('登录位置')
    plt.tight_layout()
    plt.savefig(f'{output_folder}/2-6_登录位置分布.png', dpi=300)
    plt.close()
    
    # 清理内存
    del df, location_counts
    gc.collect()
    print("登录位置分布图生成完成")

def run_analysis(output_folder=None, sample_size=None):
    """执行所有分析函数"""
    if output_folder is None:
        output_folder = create_figure_folder('new_10G_data_analysis_figure')
    else:
        output_folder = create_figure_folder(output_folder)
    
    print(f"分析结果将保存在: {output_folder}")

    # 基本人口统计学特征分析
    plot_age_distribution(output_folder, sample_size)
    plot_income_distribution(output_folder, sample_size)
    plot_geographic_distribution(output_folder, sample_size)
    plot_country_distribution(output_folder, sample_size)
    plot_age_income_heatmap(output_folder, min(50000, sample_size) if sample_size else 50000)
    plot_session_duration_by_age(output_folder, sample_size)
    
    # 购买行为和用户活动分析
    plot_device_usage(output_folder, sample_size)
    plot_purchase_categories(output_folder, sample_size)
    plot_payment_method_distribution(output_folder, sample_size)
    plot_payment_status_by_gender(output_folder, sample_size)
    plot_account_age_vs_purchases(output_folder, sample_size)
    plot_login_locations(output_folder, sample_size)
    
    return "分析完成"

# 主函数
def main():
    time_start = time.time()
    
    # 指定输出文件夹和样本大小
    result = run_analysis(output_folder='new_30G_data_analysis_figure', sample_size=None)
    print(result)
    
    time_end = time.time()
    print(f"总耗时: {time_end - time_start:.2f}秒")

if __name__ == "__main__":
    main()
