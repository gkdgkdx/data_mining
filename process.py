import pandas as pd
import json
import numpy as np
import os
from glob import glob
import time
from tqdm import tqdm
import warnings
import re
warnings.filterwarnings('ignore')

def preprocess_data(folder_path, columns=None, sample_size=None, output_folder="new_processed_data_10G"):
    """
    预处理数据文件，不进行用户聚合
    
    参数:
    folder_path: 存放parquet文件的文件夹路径
    columns: 要读取的列名列表，如果为None则读取所有列
    sample_size: 每个文件的样本大小，如果为None则读取全部数据
    output_folder: 处理后文件的输出文件夹
    """
    print("开始读取和预处理parquet文件...")
    start_time = time.time()
    
    # 定义需要的列，如果未指定则使用所有列
    if columns is None:
        columns = [
            'id', 'last_login', 'user_name', 'fullname', 'email', 
            'age', 'income', 'gender', 'country', 'address', 
            'purchase_history', 'is_active', 'registration_date', 
            'phone_number', 'login_history'
        ]
    
    # 获取所有parquet文件路径
    parquet_files = glob(os.path.join(folder_path, "*.parquet"))
    print(f"找到 {len(parquet_files)} 个parquet文件")
    
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建文件夹: {output_folder}")
    
    # 记录处理文件的路径
    processed_file_paths = []
    
    # 使用tqdm显示处理进度
    for file_path in tqdm(parquet_files, desc="处理文件进度", unit="文件"):
        file_start_time = time.time()
        file_name = os.path.basename(file_path)
        
        try:
            # 读取parquet文件
            df = pd.read_parquet(file_path, columns=columns)
            print(f"读取文件: {file_name}, 行数: {len(df)}")
            
            # 如果指定了样本大小，进行随机采样
            if sample_size and len(df) > sample_size:
                df = df.sample(sample_size, random_state=42)
                print(f"采样后行数: {len(df)}")
            
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
            
            # 1. 数据类型优化，减少内存使用
            # 整数类型转换
            for col in df.select_dtypes(include=['int']).columns:
                df[col] = pd.to_numeric(df[col], downcast='integer')
            
            # 浮点数类型转换
            for col in df.select_dtypes(include=['float']).columns:
                df[col] = pd.to_numeric(df[col], downcast='float')
            
            # 处理分类型数据
            for col in ['gender', 'country']:
                if col in df.columns:
                    df[col] = df[col].astype('category')
            
            # 2. 时间字段处理
            # 转换时间字段为datetime类型
            for col in ['last_login', 'registration_date']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            
            # 3. 解析购买历史
            if 'purchase_history' in df.columns:
                print("解析购买历史...")
                
                def parse_purchase_history(x):
                    if isinstance(x, str) and x:
                        try:
                            data = json.loads(x)
                            return {
                                'avg_price': data.get('avg_price', 0),
                                'categories': data.get('categories', ''),
                                'payment_method': data.get('payment_method', ''),
                                'payment_status': data.get('payment_status', ''),
                                'purchase_date': data.get('purchase_date', ''),
                                'items_count': len(data.get('items', []))
                            }
                        except:
                            return {
                                'avg_price': 0,
                                'categories': '',
                                'payment_method': '',
                                'payment_status': '',
                                'purchase_date': '',
                                'items_count': 0
                            }
                    return {
                        'avg_price': 0,
                        'categories': '',
                        'payment_method': '',
                        'payment_status': '',
                        'purchase_date': '',
                        'items_count': 0
                    }
                
                # 分批处理，减少内存使用
                batch_size = 10000
                purchase_data = []
                
                for i in tqdm(range(0, len(df), batch_size), desc="解析购买历史", unit="批次"):
                    batch = df.iloc[i:i+batch_size]
                    batch_data = batch['purchase_history'].apply(parse_purchase_history)
                    purchase_data.extend(batch_data)
                
                # 将解析后的数据添加到DataFrame
                purchase_df = pd.DataFrame(purchase_data)
                
                # 添加新列到原始DataFrame
                for col in purchase_df.columns:
                    df[f'purchase_{col}'] = purchase_df[col]
                
                # 删除原始的purchase_history列
                df.drop('purchase_history', axis=1, inplace=True)
                print("已删除原始purchase_history列以节省空间")
            
            # 4. 解析登录历史
            if 'login_history' in df.columns:
                print("解析登录历史...")
                
                def parse_login_history(x):
                    if isinstance(x, str) and x:
                        try:
                            data = json.loads(x)
                            return {
                                'avg_session_duration': data.get('avg_session_duration', 0),
                                'devices': "|".join(data.get('devices', [])),
                                'locations': "|".join(data.get('locations', [])),
                                'login_count': len(data.get('login_timestamps', []))
                            }
                        except:
                            return {
                                'avg_session_duration': 0,
                                'devices': '',
                                'locations': '',
                                'login_count': 0
                            }
                    return {
                        'avg_session_duration': 0,
                        'devices': '',
                        'locations': '',
                        'login_count': 0
                    }
                
                # 分批处理，减少内存使用
                batch_size = 10000
                login_data = []
                
                for i in tqdm(range(0, len(df), batch_size), desc="解析登录历史", unit="批次"):
                    batch = df.iloc[i:i+batch_size]
                    batch_data = batch['login_history'].apply(parse_login_history)
                    login_data.extend(batch_data)
                
                # 将解析后的数据添加到DataFrame
                login_df = pd.DataFrame(login_data)
                
                # 添加新列到原始DataFrame
                for col in login_df.columns:
                    df[f'login_{col}'] = login_df[col]
                
                # 删除原始的login_history列
                df.drop('login_history', axis=1, inplace=True)
                print("已删除原始login_history列以节省空间")
            
            # 5. 提取地址信息
            if 'address' in df.columns:
                print("处理地址信息...")
                initial_rows = len(df)
                
                # 提取省份信息 - 仅处理中文地址
                # 先检查地址是否包含中文
                df['is_chinese_address'] = df['address'].str.contains(r'[\u4e00-\u9fff]')
                
                # 定义函数提取省级行政区
                def extract_province(address):
                    if not isinstance(address, str) or not address:
                        return None
                        
                    # 直辖市和特别行政区列表
                    special_regions = ['北京市', '上海市', '天津市', '重庆市', '香港特别行政区', '澳门特别行政区']
                    
                    # 1. 先检查是否包含特别行政区
                    if '香港特别行政区' in address:
                        return '香港特别行政区'
                    if '澳门特别行政区' in address:
                        return '澳门特别行政区'
                    
                    # 2. 检查是否以直辖市开头
                    for city in special_regions[:4]:  # 只检查直辖市
                        if address.startswith(city):
                            return city
                        elif address.startswith(city[:-1]):  # 处理没有"市"字的情况，如"北京朝阳区"
                            return city
                    
                    # 3. 尝试提取省/自治区
                    province_match = re.search(r'^(.*?省|.*?自治区)', address)
                    if province_match:
                        return province_match.group(1)
                    
                    # 4. 尝试提取非直辖市的市
                    city_match = re.search(r'^(.*?市)(?!.*市)', address)  
                    if city_match:
                        return city_match.group(1)
                        
                    return None
                
                # 对中文地址提取省份
                df.loc[df['is_chinese_address'], 'province'] = df.loc[df['is_chinese_address'], 'address'].apply(extract_province)
                
                # 删除没有省份信息的行
                df = df.dropna(subset=['province'])
                
                
                df['province'] = df['province'].astype('category')
                
                
                df.drop('is_chinese_address', axis=1, inplace=True)
                
                #
                filtered_rows = initial_rows - len(df)
                print(f"过滤掉无省份信息的数据 {filtered_rows} 行（从 {initial_rows} 行减少到 {len(df)} 行）")
            
            # 6. 添加时间特征
            if 'last_login' in df.columns and pd.api.types.is_datetime64_any_dtype(df['last_login']):
                df['login_year'] = df['last_login'].dt.year
                df['login_month'] = df['last_login'].dt.month
                df['login_day'] = df['last_login'].dt.day
                df['login_hour'] = df['last_login'].dt.hour
                df['login_dayofweek'] = df['last_login'].dt.dayofweek
                df['login_is_weekend'] = df['login_dayofweek'].isin([5, 6]).astype('int8')
            
            if 'registration_date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['registration_date']):
                df['reg_year'] = df['registration_date'].dt.year
                df['reg_month'] = df['registration_date'].dt.month
                df['reg_day'] = df['registration_date'].dt.day
                
                # 计算账户年龄（从注册到现在的天数）
                now = pd.Timestamp.now()
                df['account_age_days'] = (now - df['registration_date']).dt.days
            
            # 7. 创建年龄组
            if 'age' in df.columns:
                df['age_group'] = pd.cut(df['age'], bins=[0, 18, 25, 35, 45, 55, 65, 100], 
                              labels=['<18', '18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
                df['age_group'] = df['age_group'].astype('category')
            
            # 8. 创建收入组
            if 'income' in df.columns:
                df['income_group'] = pd.qcut(df['income'].clip(lower=0), q=5, 
                                  labels=['极低', '低', '中', '高', '极高'])
                df['income_group'] = df['income_group'].astype('category')
            
            # 计算内存使用情况
            mem_usage_before = df.memory_usage(deep=True).sum() / (1024 * 1024)
            
            # 保存处理后的数据
            output_file = os.path.join(output_folder, f"processed_{file_name}")
            df.to_parquet(output_file, index=False)
            processed_file_paths.append(output_file)
            
            file_end_time = time.time()
            print(f"文件处理完成，耗时: {file_end_time - file_start_time:.2f}秒")
            print(f"处理后文件大小: {mem_usage_before:.2f} MB")
            
            # 清理内存
            del df
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {str(e)}")
            continue
    
    end_time = time.time()
    print(f"\n所有文件处理完成，总耗时: {end_time - start_time:.2f}秒")
    print(f"成功处理的文件数量: {len(processed_file_paths)}/{len(parquet_files)}")
    
    return processed_file_paths

if __name__ == "__main__":
    # 指定数据文件路径
    folder_path = "D:\\mylearn\\data_mining\\new_30G_data"
    
    # 开始处理
    start_time = time.time()
    try:
        processed_files = preprocess_data(
            folder_path=folder_path,
            columns=None,  # 读取所有列
            sample_size=None,  # 不采样，处理全部数据
            output_folder="new_processed_data_30G"
        )
        
        print(f"已处理 {len(processed_files)} 个文件")
        
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        
    end_time = time.time()
    print(f"总耗时: {(end_time - start_time) / 60:.2f}分钟")
