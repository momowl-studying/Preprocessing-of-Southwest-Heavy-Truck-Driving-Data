import pandas as pd

# 定义一个函数来检查当前行是否满足异常条件
def is_anomaly(row, data, window=3):
    index = row.name
    # 检查当前数据值是否不为0
    if row['车速'] == 0:
        return False
    # 检查梯度是否大于12
    if row['脉冲梯度'] <= 12:
        return False
    # 检查前3个数据值是否都为0
    start_pre = index - window
    end_pre = index
    window_pre = data.iloc[start_pre:end_pre]
    if window_pre.shape[0] < window:
        return False
    if not (window_pre['车速'] == 0).all():
        return False
    # 检查后3个数据值是否都为0
    start_post = index + 1
    end_post = index + window + 1
    window_post = data.iloc[start_post:end_post]
    if window_post.shape[0] < window:
        return False
    if not (window_post['车速'] == 0).all():
        return False
    # 如果所有条件都满足，则认为是异常
    return True

def main():
    # 读取Excel文件
    # file_path = 'E:/recognition/实车数据/工况数据集1.xlsx'
    file_path = 'E:/recognition/实车数据/工况数据集2.xlsx'  # 替换为你的Excel文件路径
    sheet_name = 'Sheet1'  # 替换为你的工作表名称
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # 计算梯度：下一个数据值减去当前数据值
    df['脉冲梯度'] = df['车速'] - df['车速'].shift(1)

    # 应用函数识别异常数据，apply用于对DataFrame的元素应用一个is_anomaly函数
    df['异常'] = df.apply(is_anomaly, axis=1, data=df)

    # 替换异常数据为0
    df.loc[df['异常'], '车速'] = 0

    # 删除辅助列
    df = df.drop(['脉冲梯度', '异常'], axis=1)

    # 保存处理后的数据
    # new_file_path = 'E:/recognition/实车数据/脉冲处理1.xlsx'
    new_file_path = 'E:/recognition/实车数据/脉冲处理2.xlsx'
    df.to_excel(new_file_path, index=False)

    print("异常数据已替换并保存成功！")

if __name__ == "__main__":
    main()