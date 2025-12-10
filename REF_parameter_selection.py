import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# 1. 读取 Excel 数据
def read_excel_data(file_path, sheet_name=0):
    """
    读取 Excel 文件中的数据。
    :param file_path: Excel 文件路径
    :param sheet_name: 工作表名称或索引
    :return: pandas DataFrame
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    print(f"数据读取成功，数据集包含 {df.shape[0]} 行和 {df.shape[1]} 列。")

    return df

# 2. 数据预处理
def preprocess_data(df, target_column):
    """
    分离特征和目标标签，并进行标签编码。
    :param df: 原始数据 DataFrame
    :param target_column: 目标标签列名
    :return: 特征矩阵 X 和目标向量 y
   """
    # 分离特征和目标
    X = df.drop(columns=[target_column])
    y = df[target_column]
    # 标签编码目标变量（如果为分类问题）
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_
    print(f"目标变量编码为: {class_names}")
    # 标签编码分类特征（如果有）
    # 假设所有非数值列都是分类特征
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        X[categorical_cols] = X[categorical_cols].apply(lambda col: pd.Categorical(col).codes)
        print(f"编码的分类特征列: {categorical_cols.tolist()}")

    return X, y_encoded

# 3. 定义 REF 函数
def recursive_feature_elimination_with_cv(X, y, model, param_range, cv, scoring='accuracy'):
    """
    使用 RFE 进行递归特征消除，并使用交叉验证评估每个 n_features_to_select 下的模型性能。

    :param X: 特征矩阵
    :param y: 目标向量
    :param model: 基础模型
    :param param_range: n_features_to_select 的范围
    :param cv: 交叉验证生成器
    :param scoring: 评估指标
    :return: 最佳 n_features_to_select 和对应的平均得分
    """
    results = {}
    for n_features in param_range:
        rfe = RFE(estimator=model, n_features_to_select=n_features, step=1)
        X_rfe = rfe.fit_transform(X, y)
        # 使用交叉验证评估模型性能
        scores = cross_val_score(model, X_rfe, y, cv=cv, scoring=scoring)
        mean_score = np.mean(scores)
        results[n_features] = mean_score
        print(f'n_features_to_select: {n_features}, 平均 {scoring}: {mean_score:.4f}')    # 保留四位小数点

    # 选择最佳 n_features_to_select
    best_n_features = max(results, key=results.get)
    best_score = results[best_n_features]
    print(f'最佳 n_features_to_select: {best_n_features}, 对应的平均 {scoring}: {best_score:.4f}')

    return best_n_features, best_score, results

# 4. 主函数
def main():
    import matplotlib.pyplot as plt

    # 设置全局字体为支持中文的字体，例如黑体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 文件路径
    file_path = 'E:/recognition/实车数据/标准和波动特征片段划分.xlsx'  # 请替换为您的 Excel 文件路径
    target_column = '聚类标签'  # 请替换为您的聚类标签列名
    columns_to_exclude = ['运行时间', '行驶时间']   # 需要删除的列
    # 读取数据
    df = read_excel_data(file_path)
    # 数据预处理
    X, y = preprocess_data(df.drop(columns=columns_to_exclude), target_column)
    # 定义基础模型
    # 这里以随机森林为例，您可以根据需要选择其他模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    # 定义交叉验证策略
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # 定义参数范围
    param_range = range(1, 26)  # n_features_to_select 从1到31
    # 执行 RFE 并进行交叉验证
    results = {}
    best_n_features, best_score, results = recursive_feature_elimination_with_cv(X, y, model, param_range, cv, scoring='accuracy')
    # 输出结果
    print("最终选择的特征数量:")
    print(best_n_features)
    # 可视化交叉验证结果
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, [results[n] for n in param_range], marker='o')
    plt.xlabel('n_features_to_select')
    plt.ylabel('平均准确率')
    plt.title('交叉验证结果')
    plt.grid()
    plt.show()
    # 如果您希望使用选定的特征进行进一步分析，可以创建新的 DataFrame
    # 重新进行 RFE 以获取选定的特征
    rfe = RFE(estimator=model, n_features_to_select=best_n_features, step=1)
    X_rfe = rfe.fit_transform(X, y)
    selected_features = X.columns[rfe.support_]
    print("最终选择的特征:")
    print(selected_features.tolist())
    # 创建新的 DataFrame
    selected_df = df[selected_features.tolist() + [target_column]]
    print("选定的特征及目标标签:")
    print(selected_df.head())
    # 保存结果到新的 Excel 文件
    selected_df.to_excel('E:/recognition/实车数据/最佳特征选择集合.xlsx', index=False)
    print("结果已保存到 'E:/recognition/实车数据/最佳特征选择集合.xlsx'。")


if __name__ == "__main__":
    main()
