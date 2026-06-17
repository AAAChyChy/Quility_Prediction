import pandas as pd
import numpy as np
import os


def clip_outliers_iqr(df, factor=1.5):
    """使用 IQR 方法检测数值列异常值，并将越界值拉回上下边界。"""
    result = df.copy()
    numeric_cols = result.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        q1 = result[col].quantile(0.25)
        q3 = result[col].quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            continue

        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        result[col] = result[col].clip(lower=lower_bound, upper=upper_bound)

    return result


# 获取当前脚本所在文件夹的绝对路径
current_path = os.path.dirname(os.path.abspath(__file__))
# 获取上原始数据文件夹的绝对路径
parent_path = os.path.join(current_path, "2509010001-2606142359-已检测")
file_path = os.path.join(parent_path, "表面裂纹.csv")
# 读取数据
Data = pd.read_csv(file_path, low_memory=False)
Data = clip_outliers_iqr(Data)
print(Data.head())
