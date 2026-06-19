import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from itertools import combinations
import random

# import torch
# # 固定所有随机数生成器
random.seed(42)
# np.random.seed(42)
# torch.manual_seed(42)

# 获取当前脚本所在文件夹的绝对路径
current_path = os.path.dirname(os.path.abspath(__file__))
# 获取上原始数据文件夹的绝对路径
parent_path = os.path.join(current_path, "数据")
file_path = os.path.join(parent_path, "Data_V0.csv")
# 读取数据
Data = pd.read_csv(file_path, low_memory=False)


############################################# 数据清洗#####################################
def clip_outliers_iqr(df, columns, factor=1.5):
    """对指定数值列使用 IQR 方法，将越界值拉回上下边界。"""
    result = df.copy()

    for col in columns:
        if col not in result.columns:
            print(f"列 {col} 不存在，已跳过")
            continue

        q1 = result[col].quantile(0.25)
        q3 = result[col].quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            continue

        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr

        result[col] = result[col].clip(lower=lower_bound, upper=upper_bound)

    return result


Data_features = Data.iloc[:, 3:145]
Data_labels = Data.iloc[:, 145:150]
num_columns = [
    "大包氩气流量",
    "大包氩气压力",
    "中间包内钢水温度",
    "包盖氩气流量",
    "包盖氩气压力",
    "上水口氩气流量",
    "上水口氩气压力",
    "塞棒氩气流量",
    "塞棒氩气压力",
    "滑板氩气流量",
    "滑板氩气压力",
    "结晶器过钢量",
    "结晶器实际液面",
    "结晶器液面波动值",
    "结晶器宽度",
    "结晶器锥度",
    "电磁搅拌电流",
    "电磁搅拌频率",
    "振动单元实际频率",
    "振动单元实际振幅",
    "结晶器冷却水温差左窄",
    "结晶器冷却水温差右窄",
    "结晶器冷却水温差内弧",
    "结晶器冷却水温差外弧",
    "结晶器冷却水流量左窄",
    "结晶器冷却水流量右窄",
    "结晶器冷却水流量内弧",
    "结晶器冷却水流量外弧",
    "结晶器拉速变化率",
    "结晶器最小拉速",
    "结晶器最大拉速",
    "拉坯长度",
    "二冷压缩空气总管压力",
    "二冷水总管温度",
    "二冷水总管压力",
    "一区弧宽面实际水量",
    "一区窄面实际水量",
    "二区内外弧中部实际水量",
    "二区内外弧边部实际水量",
    "三区内外弧中部实际水量",
    "三区内外弧边部实际水量",
    "四区内外弧中部实际水量",
    "四区内外弧边部实际水量",
    "五区内外弧中部实际水量",
    "五区内外弧边部实际水量",
    "六区内弧中部实际水量",
    "六区内弧边部实际水量",
    "六区外弧中部实际水量",
    "六区外弧边部实际水量",
    "七区内弧中部实际水量",
    "七区内弧边部实际水量",
    "七区外弧中部实际水量",
    "七区外弧边部实际水量",
    "八区内弧中部实际水量",
    "八区内弧边部实际水量",
    "八区外弧中部实际水量",
    "八区外弧边部实际水量",
    "九区内弧中部实际水量",
    "九区内弧边部实际水量",
    "九区外弧中部实际水量",
    "九区外弧边部实际水量",
    "十区内弧中部实际水量",
    "十区内弧边部实际水量",
    "十一区内弧中部实际水量",
    "十一区内弧边部实际水量",
    "二三区二冷气中部实际值",
    "二三区二冷气边部实际值",
    "四区二冷气中部实际值",
    "四区二冷气边部实际值",
    "五区二冷气中部实际值",
    "五区二冷气边部实际值",
    "六区二冷气中部实际值",
    "六区二冷气边部实际值",
    "七区二冷气中部实际值",
    "七区二冷气边部实际值",
    "八区二冷气中部实际值",
    "八区二冷气边部实际值",
    "九区二冷气中部实际值",
    "九区二冷气边部实际值",
    "十区二冷气实际值",
    "十一区二冷气实际值",
    "轻压下起点",
    "轻压下终点",
    "轻压下总量",
    "平均拉速0to1m",
    "平均拉速1to2m",
    "平均拉速2to3m",
    "平均拉速3to4m",
    "平均拉速4to5m",
    "平均拉速5to6m",
    "平均拉速6to7m",
    "平均拉速7to8m",
    "平均拉速8to9m",
    "平均拉速9to10m",
    "平均拉速10to11m",
    "平均拉速11to12m",
    "平均拉速12to13m",
    "平均拉速13to14m",
    "平均拉速14to15m",
    "平均拉速15to16m",
    "平均拉速16to17m",
    "平均拉速17to18m",
    "平均拉速18to19m",
    "平均拉速19to20m",
    "平均拉速20to21m",
    "平均拉速21to22m",
    "平均拉速22to23m",
    "平均拉速23to24m",
    "平均拉速24to25m",
    "平均拉速25to26m",
    "平均拉速26to27m",
    "平均拉速27to28m",
    "平均拉速28to29m",
    "平均拉速29to30m",
    "平均拉速30to31m",
    "平均拉速31to32m",
    "平均拉速32to33m",
    "平均拉速33to34m",
    "平均拉速34to35m",
]
# 特征对数字特征进行异常值处理
Data_features = clip_outliers_iqr(Data_features, num_columns)
############################################# 特征与标签#####################################
# 连续特征标准化数据, 将缺失值设置为0
Data_features[num_columns] = Data_features[num_columns].apply(
    lambda x: (x - x.mean()) / (x.std())
)
Data_features[num_columns] = Data_features[num_columns].fillna(0)

# 类型特征独热编码
object_columns = [
    "所属流",
    "suppressSeg",
    "中包覆盖剂编号",
    "下渣报警状态",
    "结晶器保护渣类型",
    "换水口标记",
    "浸入式水口类型",
]
# 转成字符串，避免被当作连续数值
Data_features[object_columns] = Data_features[object_columns].astype(str)
# 独热编码
Data_features = pd.get_dummies(
    Data_features, columns=object_columns, dummy_na=False, dtype=int
)
############################################标签转成有无该缺陷的二分类####################################################

crack_labels = ["中间裂纹", "三角区裂纹", "角部裂纹"]
center_labels = ["中心偏析", "中心疏松"]
# 裂纹：0 为无缺陷，非 0 为有缺陷
Data_labels[crack_labels] = Data_labels[crack_labels].ne(0).astype(int)
# 偏析、疏松：0.5 为无缺陷，大于 0.5 为有缺陷
Data_labels[center_labels] = Data_labels[center_labels].gt(0.5).astype(int)

# 在最后面加一列板坯含有的缺陷类型
label_columns = ["中间裂纹", "三角区裂纹", "角部裂纹", "中心偏析", "中心疏松"]
# 确保标签只有 0 和 1
Data_labels[label_columns] = Data_labels[label_columns].astype(int)
# 建立缺陷组合与编号的映射
combination_to_code = {(): 0}
code_to_combination = {0: "无缺陷"}
code = 1
for combination_size in range(1, len(label_columns) + 1):
    for combination in combinations(label_columns, combination_size):
        combination_to_code[combination] = code
        code_to_combination[code] = "+".join(combination)
        code += 1


def get_defect_type(row):
    active_defects = tuple(label for label in label_columns if row[label] == 1)
    return combination_to_code[active_defects]


# 生成缺陷类型列
Data_labels["缺陷类型"] = Data_labels.apply(get_defect_type, axis=1)

data_all = pd.concat([Data_features, Data_labels], axis=1)
current_path_csv = os.path.join(parent_path, "Data_V1.csv")
data_all.to_csv(current_path_csv, index=False, encoding="utf-8-sig")

print(Data_labels.apply(pd.Series.value_counts))
print(f"Data_V1.csv 已保存到 {current_path_csv} 中")
print("缺陷类型对应关系：")
for code, defect_name in code_to_combination.items():
    print(f"{code}: {defect_name}")

print("\n各缺陷类型样本数量：")
print(Data_labels["缺陷类型"].value_counts().sort_index())

print(f"\nData_V1.csv 已保存到 {current_path_csv} 中")

##################################### 按4：1划分训练集和测试集###########################################
X_train, X_test, y_train, y_test = train_test_split(
    Data_features, Data_labels, test_size=0.2, random_state=42, shuffle=False
)
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)
train_data.to_csv(
    os.path.join(parent_path, "train_Data_V1.csv"), index=False, encoding="utf-8-sig"
)
test_data.to_csv(
    os.path.join(parent_path, "test_Data_V1.csv"), index=False, encoding="utf-8-sig"
)
print("训练集：", train_data.shape)
print("测试集：", test_data.shape)
