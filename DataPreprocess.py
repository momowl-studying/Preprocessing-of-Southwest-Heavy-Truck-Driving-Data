import pandas as pd
import numpy as np

def remove_consecutive_zeros(df, column='A', threshold=180):
    """
    删除列中连续超过指定阈值的零值行。

    参数：
    df (DataFrame): 要处理的数据框。
    column (str): 要检查的列名，默认为 'A'。
    threshold (int): 连续零值的阈值，默认为 180。

    返回：
    DataFrame: 处理后的数据框。
    """
    # 创建一个布尔系列，标记哪些行是零
    is_zero = df[column] == 0
    # 标记连续零的组，不为0则布尔值累计，为0则布尔值为0，没有累加
    group = (is_zero != is_zero.shift()).cumsum()
    # 计算每个组的大小，记录连续0的组连续了group_sizes个0
    group_sizes = is_zero.groupby(group).transform('sum')
    # 标记需要删除的行：零值且组大小超过阈值
    rows_to_delete = is_zero & (group_sizes > threshold)
    # 删除这些行
    df_cleaned = df[~rows_to_delete].reset_index(drop=True)
    return df_cleaned

def mark_adjacent_less_than_10_vectorized(df, column='A', window=10):
    """
    标记列中满足条件的值：当前值小于10且前后10个值也小于10，则标记为0。

    参数：
    df (DataFrame): 要处理的数据框。
    column (str): 要检查的列名，默认为 'A'。
    window (int): 要检查的窗口大小，前后各 window 个值。

    返回：
    DataFrame: 处理后的数据框。
    """
    df_modified = df.copy()
    col = df[column]

    # 创建一个布尔掩码，标记小于10的值
    mask = col < 10

    # 使用 rolling 检查前后 window 个值是否都小于10
    # rolling 默认是包含当前行的，所以窗口大小为 2*window + 1
    rolling_mask = mask.rolling(window=2 * window + 1, center=True).apply(lambda x: (x == True).all(), raw=True)

    # 将满足条件的值标记为0
    df_modified.loc[rolling_mask == 1, column] = 0

    return df_modified

def error_gradient(df, column='A', max_pos_grad=5, max_neg_grad=-16.5):
    """
    逐个数据点计算梯度，并根据给定的梯度范围调整梯度值。
    如果梯度超出范围，则将梯度限制在边界值，并将下一个数据点替换为当前数据点加上调整后的梯度。
    重复此过程，直到所有数据点的梯度都在指定范围内。

    参数:
    - df (pd.DataFrame): 输入的 DataFrame。
    - column (str): 要计算梯度的列名，默认为 'A'。
    - max_pos_grad (float): 正梯度的最大允许值，默认为 5。
    - max_neg_grad (float): 负梯度的最小允许值，默认为 -16.5。

    返回:
    - pd.DataFrame: 处理后的 DataFrame，包含原始数据、计算出的梯度，以及调整后的数据。
    """
    # 提取数据为 numpy 数组
    data = df[column].values.copy()
    n = len(data)

    # 初始化梯度数组
    gradient = np.empty(n, dtype=float)
    gradient.fill(np.nan)  # 初始化为 NaN

    # 创建一个副本用于调整
    adjusted_data = data.copy()

    # 逐个数据点计算梯度并调整
    for i in range(n - 1):
        # 计算当前梯度
        current_grad = adjusted_data[i + 1] - adjusted_data[i]

        # 判断梯度是否在允许范围内
        if current_grad > max_pos_grad:
            # 超过正梯度范围，调整下一个数据点
            adjusted_data[i + 1] = adjusted_data[i] + max_pos_grad
            gradient[i] = max_pos_grad
        elif current_grad < max_neg_grad:
            # 超过负梯度范围，调整下一个数据点
            adjusted_data[i + 1] = adjusted_data[i] + max_neg_grad
            gradient[i] = max_neg_grad
        else:
            # 在允许范围内，无需调整
            gradient[i] = current_grad

    # 对于最后一个数据点，梯度设为0（或其他适当的值）
    gradient[-1] = gradient[-2]  # 或者设置为0，根据需求调整

    # 将结果添加回 DataFrame
    df['gradient'] = gradient
    df['车速'] = adjusted_data

    # 重置索引，使用 drop=True 参数避免将旧索引添加为新的一列，索引不连续，后面标记小于10的函数中根据索引前后比较就会找不到对应索引。
    df.reset_index(drop=True, inplace=True)

    return df

def process_data(df, column_name='value', window=4):
    """
    处理Excel数据，检测梯度并执行线性插值。

    参数:
    - df: pandas DataFrame，包含数据。
    - column_name: 需要处理的数据列名。
    - window: 检查前后窗口大小。

    返回:
    - 处理后的DataFrame。
    """
    data = df[column_name].copy()
    n = len(data)

    # 创建一个布尔掩码，标记需要线性插值的行
    mask = np.zeros(n, dtype=bool)

    # 处理梯度低于-16.5的情况
    for i in range(1, n):
        if (data[i] - data[i - 1]) < -16.5:
            # 检查当前行之后四个值是否存在非零值
            start = i + 1
            end = min(i + window + 1, n)
            if any(data[start:end] != 0):
                # 找到最近的非零值
                for j in range(start, end):
                    if data[j] != 0:
                        # 检查当前值与最近的非零值之间是否存在零值
                        if np.any(data[i:j] == 0):
                            mask[i:j] = True
                        break

    # 处理梯度高于12的情况
    for i in range(1, n):
        if (data[i] - data[i - 1]) > 5:
            # 检查当前行之前四个值是否存在非零值
            start = max(i - window, 0)
            end = i
            if any(data[start:end] != 0):
                # 找到最近的非零值
                for j in range(end - 1, start - 1, -1):
                    if data[j] != 0:
                        # 检查当前值与最近的非零值之间是否存在零值
                        if np.any(data[j + 1:i] == 0):
                            mask[j + 1:i] = True
                        break

    # 进行线性插值
    data[mask] = np.nan
    data_interpolated = data.interpolate(method='linear')

    # 更新原始DataFrame
    df[column_name] = data_interpolated

    return df


# 示例用法
def main():
    # 读取 Excel 文件
    input_file = 'E:/recognition/实车数据/脉冲处理2.xlsx'  # 输入文件名
    output_file1 = 'E:/recognition/实车数据/工况处理2_1.xlsx'  # 输出文件名
    output_file2 = 'E:/recognition/实车数据/工况处理2_2.xlsx'
    output_file3 = 'E:/recognition/实车数据/工况处理2_3.xlsx'
    output_file4 = 'E:/recognition/实车数据/工况处理2_4.xlsx'
    # input_file = 'E:/recognition/实车数据/脉冲处理1.xlsx'  # 输入文件名
    # output_file1 = 'E:/recognition/实车数据/工况处理1_1.xlsx'  # 输出文件名
    # output_file2 = 'E:/recognition/实车数据/工况处理1_2.xlsx'
    # output_file3 = 'E:/recognition/实车数据/工况处理1_3.xlsx'
    # output_file4 = 'E:/recognition/实车数据/工况处理1_4.xlsx'

    # 读取数据，假设第一行为表头
    sheet_name = 'Sheet1'
    df = pd.read_excel(input_file,sheet_name=sheet_name)

    # 假设数据在名为 '车速' 的列中
    df_processed1 = process_data(df, column_name='车速')

    # 处理异常梯度，按梯度在正常范围的最大值处理，直至迭代到与原始数据值重复
    df_processed2 = error_gradient(df_processed1, column='车速', max_pos_grad=6, max_neg_grad=-16.5)

    # 处理数据，标记满足条件的值为0
    df_processed3 = mark_adjacent_less_than_10_vectorized(df_processed2, column='车速')

    # 处理数据，删除连续超过 180 个零的行
    df_processed4 = remove_consecutive_zeros(df_processed3, column='车速', threshold=180)

    # 保存处理后的数据到新的 Excel 文件
    df_processed1.to_excel(output_file1, index=False)
    df_processed2.to_excel(output_file2, index=False)
    df_processed3.to_excel(output_file3, index=False)
    df_processed4.to_excel(output_file4, index=False)

    print(f"处理完成，结果已分别保存到相应位置了")

if __name__ == "__main__":
    main()