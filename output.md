# 10G数据处理

    开始读取和预处理parquet文件...  
    找到 8 个parquet文件  
    创建文件夹: new_processed_data_10G  
    处理文件进度:     0%|                                                                                                                                                                                                             | 0/8 [00:00<?, ?文件/s] 读取文件: part-00000.parquet, 行数: 5625000
    过滤前性别分布:
    gender
    男      2700030
    女      2699818
    未指定     112847
    其他      112305
    Name: count, dtype: int64
    过滤掉非男女性别的数据 225152 行（从 5625000 行减少到 5399848 行）
    解析购买历史...
    解析购买历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 540/540 [00:16<00:00, 33.53批次/s]
    已删除原始purchase_history列以节省空间██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 540/540 [00:16<00:00, 34.33批次/s] 
    解析登录历史...
    解析登录历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 540/540 [00:16<00:00, 31.78批次/s]
    已删除原始login_history列以节省空间█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 540/540 [00:16<00:00, 33.52批次/s] 
    处理地址信息...
    过滤掉无省份信息的数据 1080447 行（从 5399848 行减少到 4319401 行）
    文件处理完成，耗时: 84.00秒
    处理后文件大小: 3511.03 MB
    处理文件进度:  12%|████████████████████████▋                                                                                                                                                                            | 1/8 [01:24<09:50, 84.41s/文件] 读取文件: part-00001.parquet, 行数: 5625000
    过滤前性别分布:
    gender
    男      2701048
    女      2699312
    其他      112633
    未指定     112007
    Name: count, dtype: int64
    过滤掉非男女性别的数据 224640 行（从 5625000 行减少到 5400360 行）
    解析购买历史...
    解析购买历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 541/541 [00:16<00:00, 33.24批次/s]
    已删除原始purchase_history列以节省空间██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 541/541 [00:16<00:00, 35.83批次/s]
    解析登录历史...
    解析登录历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 541/541 [00:16<00:00, 33.26批次/s]
    已删除原始login_history列以节省空间███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌ | 537/541 [00:16<00:00, 35.06批次/s]
    处理地址信息...
    过滤掉无省份信息的数据 1080977 行（从 5400360 行减少到 4319383 行）
    文件处理完成，耗时: 86.60秒
    处理后文件大小: 3511.22 MB
    处理文件进度:  25%|█████████████████████████████████████████████████▎                                                                                                                                                   | 2/8 [02:51<08:35, 85.95s/文件] 读取文件: part-00002.parquet, 行数: 5625000
    过滤前性别分布:
    gender
    男      2701279
    女      2699281
    未指定     112234
    其他      112206
    Name: count, dtype: int64
    过滤掉非男女性别的数据 224440 行（从 5625000 行减少到 5400560 行）
    解析购买历史...
    解析购买历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 541/541 [00:16<00:00, 33.23批次/s]
    已删除原始purchase_history列以节省空间████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌ | 537/541 [00:16<00:00, 34.92批次/s]
    解析登录历史...
    解析登录历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 541/541 [00:16<00:00, 32.61批次/s]
    已删除原始login_history列以节省空间███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌ | 537/541 [00:16<00:00, 33.61批次/s]
    处理地址信息...
    过滤掉无省份信息的数据 1081542 行（从 5400560 行减少到 4319018 行）
    文件处理完成，耗时: 87.42秒
    处理后文件大小: 3510.90 MB
    处理文件进度:  38%|█████████████████████████████████████████████████████████████████████████▉                                                                                                                           | 3/8 [04:19<07:14, 86.85s/文件] 读取文件: part-00003.parquet, 行数: 5625000
    过滤前性别分布:
    gender
    男      2700809
    女      2698982
    未指定     112762
    其他      112447
    Name: count, dtype: int64
    过滤掉非男女性别的数据 225209 行（从 5625000 行减少到 5399791 行）
    解析购买历史...
    解析购买历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 540/540 [00:16<00:00, 32.61批次/s]
    已删除原始purchase_history列以节省空间██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 540/540 [00:16<00:00, 34.30批次/s]
    解析登录历史...
    解析登录历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 540/540 [00:16<00:00, 32.52批次/s]
    已删除原始login_history列以节省空间███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉ | 537/540 [00:16<00:00, 34.35批次/s]
    处理地址信息...
    过滤掉无省份信息的数据 1080974 行（从 5399791 行减少到 4318817 行）
    文件处理完成，耗时: 87.76秒
    处理后文件大小: 3510.55 MB
    处理文件进度:  50%|██████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                                  | 4/8 [05:47<05:49, 87.40s/文件] 读取文件: part-00004.parquet, 行数: 5625000
    过滤前性别分布:
    gender
    男      2700786
    女      2699662
    其他      112472
    未指定     112080
    Name: count, dtype: int64
    过滤掉非男女性别的数据 224552 行（从 5625000 行减少到 5400448 行）
    解析购买历史...
    解析购买历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 541/541 [00:16<00:00, 32.03批次/s]
    已删除原始purchase_history列以节省空间████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌ | 537/541 [00:16<00:00, 34.49批次/s]
    解析登录历史...
    解析登录历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 541/541 [00:16<00:00, 32.85批次/s]
    已删除原始login_history列以节省空间███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌ | 537/541 [00:16<00:00, 34.10批次/s]
    处理地址信息...
    过滤掉无省份信息的数据 1080586 行（从 5400448 行减少到 4319862 行）
    文件处理完成，耗时: 89.93秒
    处理后文件大小: 3511.56 MB
    处理文件进度:  62%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                                         | 5/8 [07:18<04:25, 88.48s/文件] 读取文件: part-00005.parquet, 行数: 5625000
    过滤前性别分布:
    gender
    女      2700951
    男      2699663
    未指定     112228
    其他      112158
    Name: count, dtype: int64
    过滤掉非男女性别的数据 224386 行（从 5625000 行减少到 5400614 行）
    解析购买历史...
    解析购买历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 541/541 [00:16<00:00, 32.66批次/s]
    已删除原始purchase_history列以节省空间████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌ | 537/541 [00:16<00:00, 34.89批次/s]
    解析登录历史...
    解析登录历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 541/541 [00:16<00:00, 32.85批次/s]
    已删除原始login_history列以节省空间███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌ | 537/541 [00:16<00:00, 33.47批次/s]
    处理地址信息...
    过滤掉无省份信息的数据 1079683 行（从 5400614 行减少到 4320931 行）
    文件处理完成，耗时: 88.41秒
    处理后文件大小: 3512.42 MB
    处理文件进度:  75%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                 | 6/8 [08:46<02:57, 88.63s/文件] 读取文件: part-00006.parquet, 行数: 5625000
    过滤前性别分布:
    gender
    女      2699826
    男      2699651
    未指定     112873
    其他      112650
    Name: count, dtype: int64
    过滤掉非男女性别的数据 225523 行（从 5625000 行减少到 5399477 行）
    解析购买历史...
    解析购买历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 540/540 [00:16<00:00, 32.36批次/s]
    已删除原始purchase_history列以节省空间█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋| 539/540 [00:16<00:00, 34.23批次/s]
    解析登录历史...
    解析登录历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 540/540 [00:16<00:00, 32.81批次/s]
    已删除原始login_history列以节省空间███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉ | 537/540 [00:16<00:00, 33.23批次/s]
    处理地址信息...
    过滤掉无省份信息的数据 1079478 行（从 5399477 行减少到 4319999 行）
    文件处理完成，耗时: 88.62秒
    处理后文件大小: 3511.48 MB
    处理文件进度:  88%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                        | 7/8 [10:16<01:28, 88.78s/文件] 读取文件: part-00007.parquet, 行数: 5625000
    过滤前性别分布:
    gender
    女      2700254
    男      2700131
    未指定     112621
    其他      111994
    Name: count, dtype: int64
    过滤掉非男女性别的数据 224615 行（从 5625000 行减少到 5400385 行）
    解析购买历史...
    解析购买历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 541/541 [00:24<00:00, 22.22批次/s]
    已删除原始purchase_history列以节省空间██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 541/541 [00:24<00:00, 10.58批次/s]
    解析登录历史...
    解析登录历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 541/541 [00:18<00:00, 28.51批次/s]
    已删除原始login_history列以节省空间████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋| 540/541 [00:18<00:00, 32.53批次/s]
    处理地址信息...
    过滤掉无省份信息的数据 1079511 行（从 5400385 行减少到 4320874 行）
    文件处理完成，耗时: 110.08秒
    处理后文件大小: 3512.45 MB
    处理文件进度: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [12:06<00:00, 90.83s/文件]

    所有文件处理完成，总耗时: 726.60秒
    成功处理的文件数量: 8/8
    已处理 8 个文件
    总耗时: 12.16分钟

---

# 30G数据处理
    开始读取和预处理parquet文件...
    找到 16 个parquet文件
    创建文件夹: new_processed_data_30G
    处理文件进度:   0%|                                                                                                                                                                                                            | 0/16 [00:00<?, ?文件/s] 读取文件: part-00000.parquet, 行数: 8437500
    过滤前性别分布:
    gender
    男      4052706
    女      4047433
    未指定     169303
    其他      168058
    Name: count, dtype: int64
    过滤掉非男女性别的数据 337361 行（从 8437500 行减少到 8100139 行）
    解析购买历史...
    解析购买历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 811/811 [00:27<00:00, 29.48批次/s]
    已删除原始purchase_history列以节省空间█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊| 810/811 [00:27<00:00, 33.45批次/s] 
    解析登录历史...
    解析登录历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 811/811 [00:33<00:00, 24.43批次/s]
    已删除原始login_history列以节省空间████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌| 809/811 [00:33<00:00, 14.66批次/s] 
    处理地址信息...
    过滤掉无省份信息的数据 1618679 行（从 8100139 行减少到 6481460 行）
    文件处理完成，耗时: 198.70秒
    处理后文件大小: 5268.54 MB
    处理文件进度:   6%|████████████▏                                                                                                                                                                                      | 1/16 [03:20<50:03, 200.23s/文件] 读取文件: part-00001.parquet, 行数: 8437500
    过滤前性别分布:
    gender
    男      4050522
    女      4048987
    未指定     169104
    其他      168887
    Name: count, dtype: int64
    过滤掉非男女性别的数据 337991 行（从 8437500 行减少到 8099509 行）
    解析购买历史...
    解析购买历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 810/810 [00:44<00:00, 18.39批次/s]
    已删除原始purchase_history列以节省空间█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌| 808/810 [00:43<00:00, 34.28批次/s]
    解析登录历史...
    解析登录历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 810/810 [00:24<00:00, 33.25批次/s]
    已删除原始login_history列以节省空间█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 810/810 [00:24<00:00, 35.33批次/s]
    处理地址信息...
    过滤掉无省份信息的数据 1621579 行（从 8099509 行减少到 6477930 行）
    文件处理完成，耗时: 220.38秒
    处理后文件大小: 5265.65 MB
    处理文件进度:  12%|████████████████████████▍                                                                                                                                                                          | 2/16 [07:01<49:34, 212.47s/文件] 读取文件: part-00002.parquet, 行数: 8437500
    过滤前性别分布:
    gender
    男      4052760
    女      4047398
    未指定     168914
    其他      168428
    Name: count, dtype: int64
    过滤掉非男女性别的数据 337342 行（从 8437500 行减少到 8100158 行）
    解析购买历史...
    解析购买历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 811/811 [00:35<00:00, 22.98批次/s]
    已删除原始purchase_history列以节省空间█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ | 807/811 [00:35<00:00, 34.19批次/s]
    解析登录历史...
    解析登录历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 811/811 [00:25<00:00, 31.20批次/s]
    已删除原始login_history列以节省空间████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊| 810/811 [00:25<00:00, 33.88批次/s]
    处理地址信息...
    过滤掉无省份信息的数据 1618570 行（从 8100158 行减少到 6481588 行）
    文件处理完成，耗时: 175.95秒
    处理后文件大小: 5268.83 MB
    处理文件进度:  19%|████████████████████████████████████▌                                                                                                                                                              | 3/16 [09:58<42:31, 196.29s/文件] 读取文件: part-00003.parquet, 行数: 8437500
    过滤前性别分布:
    gender
    男      4050061
    女      4050007
    其他      169311
    未指定     168121
    Name: count, dtype: int64
    过滤掉非男女性别的数据 337432 行（从 8437500 行减少到 8100068 行）
    解析购买历史...
    解析购买历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 811/811 [00:29<00:00, 27.53批次/s]
    已删除原始purchase_history列以节省空间█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎| 808/811 [00:29<00:00, 33.53批次/s]
    解析登录历史...
    解析登录历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 811/811 [00:25<00:00, 31.90批次/s]
    已删除原始login_history列以节省空间████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊| 810/811 [00:25<00:00, 34.17批次/s]
    处理地址信息...
    过滤掉无省份信息的数据 1619706 行（从 8100068 行减少到 6480362 行）
    文件处理完成，耗时: 165.28秒
    处理后文件大小: 5267.74 MB
    处理文件进度:  25%|████████████████████████████████████████████████▊                                                                                                                                                  | 4/16 [12:44<36:51, 184.32s/文件] 读取文件: part-00004.parquet, 行数: 8437500
    过滤前性别分布:
    gender
    男      4053177
    女      4047083
    未指定     168639
    其他      168601
    Name: count, dtype: int64
    过滤掉非男女性别的数据 337240 行（从 8437500 行减少到 8100260 行）
    解析购买历史...
    解析购买历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 811/811 [00:29<00:00, 27.51批次/s]
    已删除原始purchase_history列以节省空间█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊| 810/811 [00:29<00:00, 33.64批次/s]
    解析登录历史...
    解析登录历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 811/811 [00:25<00:00, 31.44批次/s]
    已删除原始login_history列以节省空间████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊| 810/811 [00:25<00:00, 34.68批次/s]
    处理地址信息...
    过滤掉无省份信息的数据 1619181 行（从 8100260 行减少到 6481079 行）
    文件处理完成，耗时: 169.80秒
    处理后文件大小: 5268.28 MB
    处理文件进度:  31%|████████████████████████████████████████████████████████████▉                                                                                                                                      | 5/16 [15:35<32:54, 179.54s/文件] 读取文件: part-00005.parquet, 行数: 8437500
    过滤前性别分布:
    gender
    男      4050746
    女      4049484
    未指定     168763
    其他      168507
    Name: count, dtype: int64
    过滤掉非男女性别的数据 337270 行（从 8437500 行减少到 8100230 行）
    解析购买历史...
    解析购买历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 811/811 [00:32<00:00, 24.73批次/s]
    已删除原始purchase_history列以节省空间█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌| 809/811 [00:32<00:00, 33.13批次/s]
    解析登录历史...
    解析登录历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 811/811 [00:25<00:00, 32.10批次/s]
    已删除原始login_history列以节省空间████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌| 809/811 [00:25<00:00, 34.37批次/s]
    处理地址信息...
    过滤掉无省份信息的数据 1619291 行（从 8100230 行减少到 6480939 行）
    文件处理完成，耗时: 184.23秒
    处理后文件大小: 5268.20 MB
    处理文件进度:  38%|█████████████████████████████████████████████████████████████████████████▏                                                                                                                         | 6/16 [18:40<30:14, 181.46s/文件] 读取文件: part-00006.parquet, 行数: 8437500
    过滤前性别分布:
    gender
    男      4051858
    女      4048636
    其他      168536
    未指定     168470
    Name: count, dtype: int64
    过滤掉非男女性别的数据 337006 行（从 8437500 行减少到 8100494 行）
    解析购买历史...
    解析购买历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 811/811 [00:33<00:00, 23.93批次/s]
    已删除原始purchase_history列以节省空间█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊| 810/811 [00:33<00:00, 33.47批次/s]
    解析登录历史...
    解析登录历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 811/811 [00:26<00:00, 31.01批次/s]
    已删除原始login_history列以节省空间████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎| 808/811 [00:26<00:00, 33.28批次/s]
    处理地址信息...
    过滤掉无省份信息的数据 1619197 行（从 8100494 行减少到 6481297 行）
    文件处理完成，耗时: 182.19秒
    处理后文件大小: 5268.49 MB
    处理文件进度:  44%|█████████████████████████████████████████████████████████████████████████████████████▎                                                                                                             | 7/16 [21:43<27:18, 182.05s/文件] 读取文件: part-00007.parquet, 行数: 8437500
    过滤前性别分布:
    gender
    女      4052137
    男      4047836
    其他      168834
    未指定     168693
    Name: count, dtype: int64
    过滤掉非男女性别的数据 337527 行（从 8437500 行减少到 8099973 行）
    解析购买历史...
    解析购买历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 810/810 [00:32<00:00, 25.10批次/s]
    已删除原始purchase_history列以节省空间█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊| 809/810 [00:32<00:00, 33.53批次/s]
    解析登录历史...
    解析登录历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 810/810 [00:25<00:00, 32.17批次/s]
    已删除原始login_history列以节省空间████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌| 808/810 [00:25<00:00, 33.87批次/s]
    处理地址信息...
    过滤掉无省份信息的数据 1619736 行（从 8099973 行减少到 6480237 行）
    文件处理完成，耗时: 196.07秒
    处理后文件大小: 5267.65 MB
    处理文件进度:  50%|█████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                                 | 8/16 [25:00<24:54, 186.78s/文件] 读取文件: part-00008.parquet, 行数: 8437500
    过滤前性别分布:
    gender
    女      4051591
    男      4048506
    其他      169253
    未指定     168150
    Name: count, dtype: int64
    过滤掉非男女性别的数据 337403 行（从 8437500 行减少到 8100097 行）
    解析购买历史...
    解析购买历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 811/811 [00:31<00:00, 25.67批次/s]
    已删除原始purchase_history列以节省空间█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ | 807/811 [00:31<00:00, 31.17批次/s]
    解析登录历史...
    解析登录历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 811/811 [00:25<00:00, 32.39批次/s]
    已删除原始login_history列以节省空间████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌| 809/811 [00:25<00:00, 33.78批次/s]
    处理地址信息...
    过滤掉无省份信息的数据 1621167 行（从 8100097 行减少到 6478930 行）
    文件处理完成，耗时: 188.69秒
    处理后文件大小: 5266.51 MB
    处理文件进度:  56%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                                     | 9/16 [28:10<21:54, 187.79s/文件] 读取文件: part-00009.parquet, 行数: 8437500
    过滤前性别分布:
    gender
    女      4050256
    男      4048994
    未指定     169358
    其他      168892
    Name: count, dtype: int64
    过滤掉非男女性别的数据 338250 行（从 8437500 行减少到 8099250 行）
    解析购买历史...
    解析购买历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 810/810 [00:35<00:00, 22.54批次/s]
    已删除原始purchase_history列以节省空间██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 810/810 [00:35<00:00, 32.94批次/s]
    解析登录历史...
    解析登录历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 810/810 [00:24<00:00, 32.68批次/s]
    已删除原始login_history列以节省空间█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 810/810 [00:24<00:00, 35.39批次/s]
    处理地址信息...
    过滤掉无省份信息的数据 1618730 行（从 8099250 行减少到 6480520 行）
    文件处理完成，耗时: 193.30秒
    处理后文件大小: 5267.69 MB
    处理文件进度:  62%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                        | 10/16 [31:25<18:59, 189.89s/文件] 读取文件: part-00010.parquet, 行数: 8437500
    过滤前性别分布:
    gender
    女      4052375
    男      4048192
    其他      168544
    未指定     168389
    Name: count, dtype: int64
    过滤掉非男女性别的数据 336933 行（从 8437500 行减少到 8100567 行）
    解析购买历史...
    解析购买历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 811/811 [00:35<00:00, 22.69批次/s]
    已删除原始purchase_history列以节省空间█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊| 810/811 [00:35<00:00, 26.86批次/s]
    解析登录历史...
    解析登录历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 811/811 [00:26<00:00, 30.90批次/s]
    已删除原始login_history列以节省空间████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌| 809/811 [00:26<00:00, 31.74批次/s]
    处理地址信息...
    过滤掉无省份信息的数据 1620472 行（从 8100567 行减少到 6480095 行）
    文件处理完成，耗时: 201.47秒
    处理后文件大小: 5267.51 MB
    处理文件进度:  69%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                            | 11/16 [34:47<16:08, 193.66s/文件] 读取文件: part-00011.parquet, 行数: 8437500
    过滤前性别分布:
    gender
    女      4050185
    男      4049890
    未指定     168868
    其他      168557
    Name: count, dtype: int64
    过滤掉非男女性别的数据 337425 行（从 8437500 行减少到 8100075 行）
    解析购买历史...
    解析购买历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 811/811 [00:32<00:00, 25.07批次/s]
    已删除原始purchase_history列以节省空间█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌| 809/811 [00:32<00:00, 30.73批次/s]
    解析登录历史...
    解析登录历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 811/811 [00:26<00:00, 30.59批次/s]
    已删除原始login_history列以节省空间████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ | 807/811 [00:26<00:00, 31.40批次/s]
    处理地址信息...
    过滤掉无省份信息的数据 1621432 行（从 8100075 行减少到 6478643 行）
    文件处理完成，耗时: 193.69秒
    处理后文件大小: 5266.32 MB
    处理文件进度:  75%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                | 12/16 [38:02<12:56, 194.19s/文件] 读取文件: part-00012.parquet, 行数: 8437500
    过滤前性别分布:
    gender
    男      4052574
    女      4048233
    未指定     169062
    其他      167631
    Name: count, dtype: int64
    过滤掉非男女性别的数据 336693 行（从 8437500 行减少到 8100807 行）
    解析购买历史...
    解析购买历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 811/811 [01:15<00:00, 10.67批次/s]
    已删除原始purchase_history列以节省空间█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊| 810/811 [01:15<00:00, 12.69批次/s]
    解析登录历史...
    解析登录历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 811/811 [01:00<00:00, 13.33批次/s]
    已删除原始login_history列以节省空间████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌| 809/811 [01:00<00:00, 13.84批次/s]
    处理地址信息...
    过滤掉无省份信息的数据 1619098 行（从 8100807 行减少到 6481709 行）
    文件处理完成，耗时: 377.39秒
    处理后文件大小: 5268.97 MB
    处理文件进度:  81%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                    | 13/16 [44:21<12:30, 250.13s/文件] 读取文件: part-00013.parquet, 行数: 8437500
    过滤前性别分布:
    gender
    男      4052790
    女      4047725
    其他      168602
    未指定     168383
    Name: count, dtype: int64
    过滤掉非男女性别的数据 336985 行（从 8437500 行减少到 8100515 行）
    解析购买历史...
    解析购买历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 811/811 [01:19<00:00, 10.21批次/s]
    已删除原始purchase_history列以节省空间█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌| 809/811 [01:19<00:00, 13.76批次/s]
    解析登录历史...
    解析登录历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 811/811 [01:03<00:00, 12.75批次/s]
    已删除原始login_history列以节省空间████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌| 809/811 [01:03<00:00, 12.95批次/s]
    处理地址信息...
    过滤掉无省份信息的数据 1620367 行（从 8100515 行减少到 6480148 行）
    文件处理完成，耗时: 396.42秒
    处理后文件大小: 5267.59 MB
    处理文件进度:  88%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                        | 14/16 [50:59<09:49, 294.73s/文件] 读取文件: part-00014.parquet, 行数: 8437500
    过滤前性别分布:
    gender
    男      4050275
    女      4049823
    其他      169265
    未指定     168137
    Name: count, dtype: int64
    过滤掉非男女性别的数据 337402 行（从 8437500 行减少到 8100098 行）
    解析购买历史...
    解析购买历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 811/811 [01:11<00:00, 11.29批次/s]
    已删除原始purchase_history列以节省空间█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌| 809/811 [01:11<00:00, 13.63批次/s]
    解析登录历史...
    解析登录历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 811/811 [01:04<00:00, 12.56批次/s]
    已删除原始login_history列以节省空间████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌| 809/811 [01:04<00:00, 13.12批次/s]
    处理地址信息...
    过滤掉无省份信息的数据 1618826 行（从 8100098 行减少到 6481272 行）
    文件处理完成，耗时: 390.58秒
    处理后文件大小: 5268.42 MB
    处理文件进度:  94%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉            | 15/16 [57:31<05:24, 324.12s/文件] 读取文件: part-00015.parquet, 行数: 8437500
    过滤前性别分布:
    gender
    女      4051210
    男      4049614
    其他      168466
    未指定     168210
    Name: count, dtype: int64
    过滤掉非男女性别的数据 336676 行（从 8437500 行减少到 8100824 行）
    解析购买历史...
    解析购买历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 811/811 [01:17<00:00, 10.46批次/s]
    已删除原始purchase_history列以节省空间█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌| 809/811 [01:17<00:00, 12.63批次/s]
    解析登录历史...
    解析登录历史: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 811/811 [01:04<00:00, 12.50批次/s]
    已删除原始login_history列以节省空间████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌| 809/811 [01:04<00:00, 12.91批次/s]
    处理地址信息...
    过滤掉无省份信息的数据 1619371 行（从 8100824 行减少到 6481453 行）
    文件处理完成，耗时: 384.04秒
    处理后文件大小: 5268.72 MB
    处理文件进度: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [1:03:57<00:00, 239.83s/文件]

    所有文件处理完成，总耗时: 3837.31秒
    成功处理的文件数量: 16/16
    已处理 16 个文件
    总耗时: 64.19分钟
