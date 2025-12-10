import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 读取 Excel 文件
    # input_file_path = 'E:/recognition/实车数据/工况处理1_4.xlsx'
    input_file_path = 'E:/recognition/实车数据/工况处理2_4.xlsx'
    # output_file_path = 'E:/recognition/实车数据/滤波处理1.xlsx'
    output_file_path = 'E:/recognition/实车数据/滤波处理2.xlsx'
    sheet_name = 'Sheet1'  # 替换为你的工作表名称

    # 读取 Excel 文件时，默认第一行作为表头
    df = pd.read_excel(input_file_path, sheet_name=sheet_name)

    # 选择要进行滤波的列，例如 '车速'
    data_column = '车速'

    # 检查列名是否存在
    if data_column not in df.columns:
        raise ValueError(f"列名 '{data_column}' 不存在于 Excel 文件中。请检查列名是否正确。")

    # 应用移动均值滤波
    window_size = 5  # 窗口大小，可以根据需要调整

    # 使用 rolling 计算移动均值，rolling是计算当前点与前面几个点组成设定的窗口大小求和取均值
    df['Moving_Avg_VehSpd'] = df[data_column].rolling(window=window_size).mean()

    # 识别由于窗口大小限制而缺失的NaN值
    # 这些位置对应于前 (window_size - 1) 行
    missing_values_mask = df['Moving_Avg_VehSpd'].isna()

    # 用原始数据填充这些缺失值
    df.loc[missing_values_mask, 'Moving_Avg_VehSpd'] = df.loc[missing_values_mask, data_column]

    # 选择要可视化的数据子集
    subset_size = 300  # 你可以根据需要调整子集大小
    subset_df = df.head(subset_size)

    # 可视化原始数据和滤波后的数据
    plt.figure(figsize=(10, 6))
    plt.plot(subset_df.index,subset_df[data_column], label='原始车速数据', marker='o')
    plt.plot(subset_df.index,subset_df['Moving_Avg_VehSpd'], label=f'{window_size}-点移动均值', marker='x')
    plt.title('温度数据的移动均值滤波')
    plt.xlabel('样本点')
    plt.ylabel('温度')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 将结果保存到新的 Excel 文件
    df.to_excel(output_file_path, index=False)
    print(f"\n滤波后的数据已保存到 {output_file_path}")

if __name__ == "__main__":
    main()