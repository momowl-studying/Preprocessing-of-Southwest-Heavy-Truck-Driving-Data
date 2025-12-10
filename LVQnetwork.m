% LVQ 模型实现示例

% 清空环境
clear; clc; close all;

%% 1. 数据加载与预处理

% 1.1 读取数据
% 假设数据保存在 'data.xlsx' 文件中，第一列到第二十列是特征，最后一列是标签
% 请根据实际情况修改文件路径和文件名
filename = 'E:/recognition/实车数据/最佳特征选择集合2.xlsx';
data = readtable(filename);

% 1.2 分离特征和标签
X = table2array(data(:, 1:9));  % 特征数据
% X = table2array(data(:, 1:20));  % 特征数据
y = table2array(data(:, end));   % 标签数据

% 1.3 将标签转换为类别向量（one-hot 编码）
num_classes = 3;
T = full(ind2vec(y', num_classes))';  % 将标签转换为 one-hot 编码

% 1.4 数据标准化
[X_norm, mu, sigma] = zscore(X);  % 标准化特征数据

% 1.5 划分训练集和测试集
train_ratio = 0.8;
num_train = floor(train_ratio * size(X_norm, 1));
X_train = X_norm(1:num_train, :)';
X_test = X_norm(num_train+1:end, :)';
T_train = T(1:num_train, :)';
T_test = T(num_train+1:end, :)';

%% 2. LVQ 模型参数初始化

% 2.1 定义原型向量的数量
% 通常，每个类别至少有一个原型向量
% 这里假设每个类别有 2 个原型向量
num_prototypes_per_class = 5;
prototypes = [];
prototype_labels = [];

for i = 1:num_classes
    % 获取当前类别的数据
    class_indices = find(y == i);
    class_data = X_norm(class_indices, :);
    
    % 随机选择原型向量
    rng('shuffle');
    selected_indices = randperm(length(class_indices), num_prototypes_per_class);
    prototypes = [prototypes; class_data(selected_indices, :)];
    prototype_labels = [prototype_labels; repmat(i, num_prototypes_per_class, 1)];
end

% 类别1是平原高速，类别二是丘陵高速，类别三是国道
prototypes_std = zscore(prototypes);


% 2.2 初始化学习率
learning_rate_initial = 0.01;
learning_rate = learning_rate_initial;

% 2.3 定义总训练轮数
epochs = 100;

%% 3. LVQ 模型训练

% LVQ1
% for epoch = 1:epochs
%     % 学习率衰减
%     learning_rate = learning_rate_initial * (1 - epoch / epochs);
%     
%     % 打乱训练数据顺序
%     shuffled_indices = randperm(size(X_train, 2));
%     X_train_shuffled = X_train(:, shuffled_indices);
%     %T_train = T_train'; %先转置，保证行数是样本数才能执行后面按行索引打乱
%     T_train_shuffled = T_train(:, shuffled_indices);    %最后的'是转置操作符的意思
%     
%     for i = 1:size(X_train_shuffled, 2) % size(X_train_shuffled, 2)返回里面对应变量的列数，1是行数
%         % 获取当前样本，x记录每一个样本特征，维度20x1,转置改为1x20，与原型向量维度6x20基本一致
%         x = X_train_shuffled(:, i)';
%         t = T_train_shuffled(:, i);
%         
%         % 计算距离,sqrt(x,2)代表对x沿着列求和再开方，符合求x与每个原型向量的距离
%         distances = sqrt(sum((prototypes_std - x).^2, 2));
%         
%         % 找到最近的原型向量的索引，~ 表示忽略最小值本身，只获取其winner_idx（索引）。
%         [~, winner_idx] = min(distances);
%         winner_label = prototype_labels(winner_idx);
%         
%         % 获取当前样本的真实标签,one hot编码中寻找1的行索引就能获取真实标签了
%         true_label = find(t == 1);
%         
%         % 更新原型向量
%         if winner_label == true_label
%             % 如果分类正确，拉近原型向量
%             prototypes_std(winner_idx, :) = prototypes_std(winner_idx, :) + learning_rate * (x - prototypes_std(winner_idx, :));
%         else
%             % 如果分类错误，推远原型向量
%             prototypes_std(winner_idx, :) = prototypes_std(winner_idx, :) - learning_rate * (x - prototypes_std(winner_idx, :));
%         end
%     end
%     
%     % 可选：打印训练进度
%     if mod(epoch, 10) == 0 || epoch == 1
%         fprintf('Epoch %d/%d 完成\n', epoch, epochs);
%     end
%     prototypes_original = prototypes_std .* sigma + mu;
% end

% LVQ2
for epoch = 1:epochs
    % 学习率衰减
    learning_rate = learning_rate_initial * (1 - epoch / epochs);
    
    % 打乱训练数据顺序
    shuffled_indices = randperm(size(X_train, 2));
    X_train_shuffled = X_train(:, shuffled_indices);
    T_train_shuffled = T_train(:, shuffled_indices);    
    
    % 定义窗口参数 (suggested range: 0.2-0.3)
    window_size = 0.25;  % 可调整的窗口大小参数
    
    for i = 1:size(X_train_shuffled, 2)
        x = X_train_shuffled(:, i)';
        t = T_train_shuffled(:, i);
        true_label = find(t == 1);
        
        % 计算所有原型向量的距离
        distances = sqrt(sum((prototypes_std - x).^2, 2));
        
        % 找到最近的两个原型向量
        [sorted_dist, sorted_idx] = sort(distances);
        winner_idx = sorted_idx(1);
        second_idx = sorted_idx(2);
        
        winner_label = prototype_labels(winner_idx);
        second_label = prototype_labels(second_idx);
        
        % 计算两个最近距离的比值
        d1 = sorted_dist(1);
        d2 = sorted_dist(2);
        ratio = min(d1/d2, d2/d1);
        
        % LVQ2核心条件：样本落在窗口内且两个原型属于不同类
        if (winner_label ~= second_label) && (ratio > (1-window_size)/(1+window_size))
            
            % 调整最近的原型向量
            if winner_label == true_label
                prototypes_std(winner_idx, :) = prototypes_std(winner_idx, :) + learning_rate * (x - prototypes_std(winner_idx, :));
                prototypes_std(second_idx, :) = prototypes_std(second_idx, :) - learning_rate * (x - prototypes_std(second_idx, :));
            elseif second_label == true_label
                prototypes_std(winner_idx, :) = prototypes_std(winner_idx, :) - learning_rate * (x - prototypes_std(winner_idx, :));
                prototypes_std(second_idx, :) = prototypes_std(second_idx, :) + learning_rate * (x - prototypes_std(second_idx, :));
            else       % 当出现两个最近的原型向量类别都不匹配时，就将这两个原型向量都远离样本
                prototypes_std(winner_idx, :) = prototypes_std(winner_idx, :) - learning_rate * (x - prototypes_std(winner_idx, :));
                prototypes_std(second_idx, :) = prototypes_std(second_idx, :) - learning_rate * (x - prototypes_std(second_idx, :));
            end
            
        else
            % 普通LVQ1更新规则（窗口外的情况）
            if winner_label == true_label
                prototypes_std(winner_idx, :) = prototypes_std(winner_idx, :) + learning_rate * (x - prototypes_std(winner_idx, :));
            else
                prototypes_std(winner_idx, :) = prototypes_std(winner_idx, :) - learning_rate * (x - prototypes_std(winner_idx, :));
            end
        end
    end
    
    % 可选：打印训练进度
    if mod(epoch, 10) == 0 || epoch == 1
        fprintf('Epoch %d/%d 完成\n', epoch, epochs);
    end
    
    prototypes_original = prototypes_std .* sigma + mu;
end
%% 4. 模型测试与评估

% 4.1 对测试集进行预测
y_pred = zeros(size(T_test, 2), 1);

for i = 1:size(X_test, 2)
    x = X_test(:, i)';
    
    % 计算距离
    distances = sqrt(sum((prototypes_std - x).^2, 2));
    
    % 找到最近的原型向量的索引
    [~, winner_idx] = min(distances);
    y_pred(i) = prototype_labels(winner_idx);
end

% 4.2 计算准确率
accuracy = sum(y_pred == y(num_train+1:end)) / length(y_pred) * 100;
fprintf('测试集准确率: %.2f%%\n', accuracy);

% 4.3 混淆矩阵
figure;
confusionchart(y(num_train+1:end), y_pred);
title('confusion matrix');
xlabel('prediction set');
ylabel('test set');
