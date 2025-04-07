import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
import os
from glob import glob
import time
import matplotlib.colors as mcolors
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
import missingno as msno
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def read_all_parquet_files(folder_path, columns=None, sample_size=None):
    """读取文件夹中所有parquet文件并单独处理，每个源文件生成一个处理后的文件"""
    print("开始读取parquet文件...")
    start_time = time.time()
    
    # 定义必要的列，如果未指定则使用基本必要列
    if columns is None:
        # 定义基本必要列 - 根据后续分析需求选择
        columns = [
            'id', 'age', 'gender', 'income', 'credit_score', 
            'is_active', 'timestamp', 'registration_date',
            'purchase_history', 'chinese_address'
        ]
    
    # 获取所有parquet文件路径
    parquet_files = glob(os.path.join(folder_path, "*.parquet"))
    print(f"找到 {len(parquet_files)} 个parquet文件")
    
    # 创建存储处理结果的文件夹
    processed_folder = create_processed_data_folder()
    
    # 记录所有成功处理的文件路径
    processed_file_paths = []
    
    # 逐个读取并处理文件
    for i, file_path in enumerate(parquet_files):
        print(f"\n处理文件 {i+1}/{len(parquet_files)}: {os.path.basename(file_path)}")
        file_start_time = time.time()
        
        try:
            # 读取单个文件的数据
            df = pd.read_parquet(file_path, columns=columns)
            
            # 如果指定了样本大小，对读取的数据进行采样
            if sample_size and len(df) > sample_size:
                df = df.sample(sample_size, random_state=42)
                
            # 优化内存使用
            for col in df.select_dtypes(include=['int']).columns:
                df[col] = pd.to_numeric(df[col], downcast='integer')
            for col in df.select_dtypes(include=['float']).columns:
                df[col] = pd.to_numeric(df[col], downcast='float')
            
            # 转换分类数据
            for col in ['gender']:
                if col in df.columns:
                    df[col] = df[col].astype('category')
            
            # 预处理数据
            print(f"开始预处理文件: {os.path.basename(file_path)}")
            user_df = preprocess_data(df)
            
            # 保存处理后的数据
            original_file_name = os.path.basename(file_path)
            processed_file_name = f"processed_{original_file_name}"
            processed_file_path = os.path.join(processed_folder, processed_file_name)
            
            print(f"保存处理后的数据到: {processed_file_path}")
            user_df.to_parquet(processed_file_path, index=False)
            
            # 记录成功处理的文件路径
            processed_file_paths.append(processed_file_path)
            
            file_end_time = time.time()
            print(f"文件处理完成，耗时: {file_end_time - file_start_time:.2f}秒")
            print(f"原始文件大小: {df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")
            print(f"处理后文件大小: {user_df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")
            
            # 清理内存
            del df, user_df
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"处理文件时出错: {str(e)}")
            try:
                print("尝试以默认方式读取文件...")
                df = pd.read_parquet(file_path)
                
                # 只保留需要的列
                available_cols = [col for col in columns if col in df.columns]
                df = df[available_cols]
                
                # 采样
                if sample_size and len(df) > sample_size:
                    df = df.sample(sample_size, random_state=42)
                
                # 预处理数据
                user_df = preprocess_data(df)
                
                # 保存处理后的数据
                original_file_name = os.path.basename(file_path)
                processed_file_name = f"processed_{original_file_name}"
                processed_file_path = os.path.join(processed_folder, processed_file_name)
                
                print(f"保存处理后的数据到: {processed_file_path}")
                user_df.to_parquet(processed_file_path, index=False)
                
                # 记录成功处理的文件路径
                processed_file_paths.append(processed_file_path)
                
                file_end_time = time.time()
                print(f"文件处理完成，耗时: {file_end_time - file_start_time:.2f}秒")
                
                # 清理内存
                del df, user_df
                import gc
                gc.collect()
                
            except Exception as e2:
                print(f"无法处理文件: {str(e2)}")
                continue
    
    if not processed_file_paths:
        raise ValueError("没有成功处理任何数据文件")
    
    end_time = time.time()
    print(f"\n所有文件处理完成，总耗时: {end_time - start_time:.2f}秒")
    print(f"成功处理的文件数量: {len(processed_file_paths)}")
    
    # 为了保持与原函数接口一致，返回第一个处理后的文件内容和所有处理文件的路径
    result = pd.read_parquet(processed_file_paths[0])
    return result, processed_file_paths


def preprocess_data(df):
    """预处理数据，针对性处理各列以减少内存使用，并将同一用户的多条记录合并"""
    print("===== 数据预处理 =====")
    
    # 检查是否有重复的用户记录 - 只按id识别用户
    user_counts = df.groupby('id').size().reset_index(name='记录数')
    duplicate_users = user_counts[user_counts['记录数'] > 1]
    
    if len(duplicate_users) > 0:
        print(f"发现 {len(duplicate_users)} 个用户ID有多条记录，开始聚合处理...")
        print(f"聚合前记录总数: {len(df)}")
    
    # 过滤性别数据，只保留"男"和"女"
    if 'gender' in df.columns:
        initial_rows = len(df)
        gender_counts = df['gender'].value_counts()
        print(f"过滤前性别分布:\n{gender_counts}")
        
        # 只保留性别为"男"或"女"的数据
        df = df[df['gender'].isin(['男', '女'])]
        
        # 重新设置gender为只包含男女两个类别的category类型
        df['gender'] = df['gender'].astype(str)
        df['gender'] = pd.Categorical(df['gender'], categories=['男', '女'])
        
        filtered_rows = initial_rows - len(df)
        print(f"过滤掉非男女性别的数据 {filtered_rows} 行（从 {initial_rows} 行减少到 {len(df)} 行）")
    
    # 解析购买历史并为每条记录创建初步指标
    print("解析购买历史...")
    
    # 解析每条记录的购买历史 - 不再存储具体的items
    def parse_purchase(x):
        if isinstance(x, str) and x:
            try:
                data = json.loads(x)
                return {
                    'purchase_data': data,
                    'avg_price': data.get('average_price', 0),
                    'category': data.get('category', ''),
                    'quantity': len(data.get('items', [])),  # 仍然需要计算数量
                }
            except:
                pass
        return {
            'purchase_data': {},
            'avg_price': 0,
            'category': '',
            'quantity': 0,
        }
    
    # 每批处理的记录数
    batch_size = 10000
    all_parsed_data = []
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        batch_parsed = batch['purchase_history'].apply(parse_purchase)
        all_parsed_data.extend(batch_parsed)
    
    # 将解析后的数据添加到DataFrame
    parsed_df = pd.DataFrame(all_parsed_data)
    df = pd.concat([df.reset_index(drop=True), parsed_df.reset_index(drop=True)], axis=1)
    
    # 转换时间列为日期时间类型，确保时区一致性
    print("处理时间字段...")
    # 转换时间并统一移除时区信息
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    df['registration_date'] = pd.to_datetime(df['registration_date']).dt.tz_localize(None)
    
    # 计算活跃天数
    df['active_days'] = (df['timestamp'] - df['registration_date']).dt.days
    
    # 添加时间特征
    df['reg_year'] = df['registration_date'].dt.year
    df['reg_month'] = df['registration_date'].dt.month
    df['activity_hour'] = df['timestamp'].dt.hour
    df['activity_dayofweek'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['activity_dayofweek'].isin([5, 6]).astype('int8')
    
    # 按用户ID聚合数据（不再使用user_name作为分组依据）
    print("按ID聚合同一用户的多条记录...")
    
    # 定义聚合函数
    agg_functions = {
        # 基本用户信息 - 取第一条记录的值
        'user_name': 'first',  # 保留一个用户名
        'chinese_name': 'first',
        'email': 'first',
        'age': 'first',
        'income': 'first',
        'gender': 'first',
        'country': 'first',
        'chinese_address': 'first',
        'credit_score': 'first',
        'phone_number': 'first',
        
        # 注册信息 - 取最早的
        'registration_date': 'min',
        
        # 活动信息
        'timestamp': 'max',  # 最近一次活动时间
        'is_active': 'max',  # 只要有一次是活跃的就算活跃
        
        # 购买相关 - 累计
        'quantity': 'sum',  # 累计购买项目数
        'purchase_data': lambda x: list(x),  # 保存所有订单
        'category': lambda x: list(x),  # 所有订单的类别
        'avg_price': 'mean',  # 平均价格
    }
    
    # 执行聚合 - 只按id分组
    user_df = df.groupby('id').agg(agg_functions).reset_index()
    
    # 创建新的聚合指标
    user_df['order_count'] = user_df['purchase_data'].apply(len)  # 订单数量
    user_df['unique_categories'] = user_df['category'].apply(lambda x: len(set(filter(None, x))))  # 不同类别数
    
    # 计算总消费金额 (每个订单的均价乘以其中的项目数量，再累加)
    def calc_total_spent(row):
        total = 0
        for purchase in row['purchase_data']:
            if isinstance(purchase, dict):
                price = purchase.get('average_price', 0)
                qty = len(purchase.get('items', []))
                total += price * qty
        return total
    
    user_df['total_spent'] = user_df.apply(calc_total_spent, axis=1)
    
    # 计算活跃天数
    user_df['active_days'] = (user_df['timestamp'] - user_df['registration_date']).dt.days
    
    # 添加时间特征
    user_df['reg_year'] = user_df['registration_date'].dt.year
    user_df['reg_month'] = user_df['registration_date'].dt.month
    user_df['activity_hour'] = user_df['timestamp'].dt.hour
    user_df['activity_dayofweek'] = user_df['timestamp'].dt.dayofweek
    user_df['is_weekend'] = user_df['activity_dayofweek'].isin([5, 6]).astype('int8')
    
    # 分组年龄
    user_df['age_group'] = pd.cut(user_df['age'], bins=[0, 18, 25, 35, 45, 55, 65, 100], 
                       labels=['<18', '18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
    
    # 收入分组
    if 'income' in user_df.columns:
        user_df['income_group'] = pd.qcut(user_df['income'].clip(lower=0), q=5, 
                                  labels=['极低', '低', '中', '高', '极高'])
    
    # 提取地理信息
    if 'chinese_address' in user_df.columns:
        user_df['province'] = user_df['chinese_address'].str.extract(r'^(.*?省|.*?自治区|.*?市)')
    
    # 优化内存使用
    for col in user_df.select_dtypes(include=['int']).columns:
        user_df[col] = pd.to_numeric(user_df[col], downcast='integer')
    for col in user_df.select_dtypes(include=['float']).columns:
        user_df[col] = pd.to_numeric(user_df[col], downcast='float')
    
    # 转换分类数据
    for col in ['gender', 'age_group', 'income_group', 'province']:
        if col in user_df.columns:
            user_df[col] = user_df[col].astype('category')
    
    print(f"聚合后用户数: {len(user_df)}")
    print(f"减少了 {len(df) - len(user_df)} 条重复记录")
    print("数据预处理完成")
    
    return user_df

def create_processed_data_folder():
    """创建用于存放处理后数据的文件夹"""
    folder_name = 'processed_data'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"创建文件夹: {folder_name}")
    return folder_name

def save_processed_data(df, original_file_path):
    """将处理后的数据保存为parquet文件，存放在processed_data文件夹中"""
    # 创建保存数据的文件夹
    processed_folder = create_processed_data_folder()
    
    # 获取原始文件名（不含路径）
    original_file_name = os.path.basename(original_file_path)
    # 构建新文件名
    processed_file_name = f"processed_{original_file_name}"
    # 构建保存路径（保存在processed_data文件夹中）
    processed_file_path = os.path.join(processed_folder, processed_file_name)
    
    print(f"正在保存处理后的数据到 {processed_file_path}...")
    start_time = time.time()
    
    try:
        # 保存为parquet文件
        df.to_parquet(processed_file_path, index=False)
        end_time = time.time()
        print(f"数据保存成功，耗时: {end_time - start_time:.2f}秒")
        print(f"保存的文件大小: {os.path.getsize(processed_file_path) / (1024 * 1024):.2f} MB")
        return processed_file_path
    except Exception as e:
        print(f"保存数据时出错: {str(e)}")
        return None

def create_figure_folder():
    folder_name = 'code_2_analysis_figure'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"创建文件夹: {folder_name}")
    return folder_name



def run_analysis(df, file_paths):
    """运行所有分析，采用分阶段处理以减少内存峰值"""
    figure_folder = create_figure_folder()
    
    # 数据预处理 - 包含用户数据聚合
    user_df = preprocess_data(df)
    
    # 将处理后的数据保存为parquet文件
    if file_paths and len(file_paths) > 0:
        original_file_path = file_paths[0]  # 使用第一个文件路径
        processed_file_path = save_processed_data(user_df, original_file_path)
        if processed_file_path:
            print(f"处理后的数据已保存至: {processed_file_path}")

    
    return user_df

if __name__ == "__main__":
    try:
        # 指定数据文件路径
        folder_path = 'D:\\mylearn\\data_mining\\10G_data'
        
        # 读取文件，加入user_name列
        needed_columns = [
            'id', 'user_name', 'chinese_name', 'email', 
            'age', 'gender', 'income', 'credit_score', 
            'is_active', 'timestamp', 'registration_date',
            'purchase_history', 'chinese_address', 'country', 'phone_number'
        ]
        
        # 读取文件
        print("开始读取数据...")
        df, file_paths = read_all_parquet_files(folder_path, columns=needed_columns, sample_size=None)
        print(f"数据读取完成，总行数: {len(df)}")
        
        # 运行分析
        user_df = run_analysis(df, file_paths)
        
        # 打印用户数量
        print(f"总用户数量: {len(user_df)}")
        
        # 分析完成后清理内存
        del df, user_df
        import gc
        gc.collect()
        
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()