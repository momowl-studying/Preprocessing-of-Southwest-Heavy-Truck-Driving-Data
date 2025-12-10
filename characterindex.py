import pandas as pd
from openpyxl import Workbook
import numpy as np
import traceback

def split_excel_data_with_header(input_file, output_file, sheet_name=0, column_name=None):
    # 读取Excel文件，假设第一行是表头
    df = pd.read_excel(input_file, sheet_name=sheet_name, header=0)

    # 如果指定了列名，则使用列名提取数据；否则，默认使用第一列
    if column_name:
        # 选取想要的列的数据
        data = df[column_name].tolist()
    else:
        # 假设第一列是数据列
        data = df.iloc[:, 0].tolist()

    segments = []
    current_segment = []
    segment_lengths = []  # 用于记录每个数据段的长度
    v_meansegment = []
    vr_meansegment = []
    v_maxsegment = []
    a_meansegment = []
    a_maxsegment = []
    adown_meansegment = []
    adown_maxsegment = []
    idle_ratiosegment = []
    up_ratiosegment = []
    down_ratiosegment = []
    cruise_ratiosegment = []    #匀速行驶都可以视为巡航
    v_StdDevsegment = []
    p_StdDevsegment = []
    s_segment = []
    n_StdDevsegment = []
    Ratio0_10 = []
    Ratio10_20 = []
    Ratio20_30 = []
    Ratio30_40 = []
    Ratio40_50 = []
    Ratio50_60 = []
    Ratio60_70 = []
    Ratio70_80 = []
    Ratio80_90 = []
    Ratio90_100 = []
    Mmean100_2 = []
    Mmean100_10 = []
    Mmax100_2 = []
    Mmax100_10 = []
    Tmean100_2 = []
    Tmean100_10 = []
    Tmax100_2 = []
    Tmax100_10 = []
    a2_mean = []
    T_run = []


    #对数据进行遍历，当值为0时判断是否为连续出现的0，是则记录，不是则保存片段
    for value in data:
        if value == 0:
            #对于速度小于10s的片段都看作怠速，可以不用求其特征
            if sum(current_segment) != 0:  # 如果当前段不为空，则保存当前段
                if sum(1 for x in current_segment if x > 0) < 10:# 没有使用列表推导式，使用生成器表达式，可以减少存储列表的内存占用
                    current_segment = []
                    continue
                current_segment.insert(0,0)#由于判断时每一个数据段第一个0用于前面数据段的划分判断，因此需要再给数据段前面加0，但是第一个数据段就多了一个0
                segments.append(current_segment)

                # 平均速度
                v_mean = sum(current_segment)/len(current_segment)
                v_meansegment.append(v_mean)

                # 平均行驶速度
                # 使用 remove() 方法删除所有0元素
                currentrun_segment = list(current_segment)
                while 0 in currentrun_segment:
                    currentrun_segment.remove(0)
                vr_mean = sum(currentrun_segment) / len(currentrun_segment)
                vr_meansegment.append(vr_mean)

                # 行驶时间
                T_run.append(len(currentrun_segment))

                # 最大速度
                v_max = max(current_segment)
                v_maxsegment.append(v_max)

                # 初始化一个与 data 长度相同的空列表，用于存储梯度
                gradient = [0.0] * len(current_segment)  # 初始化梯度列表，所有元素初始为 0.0

                # 计算前 n-1 个梯度
                for i in range(len(current_segment) - 1):
                    gradient[i] = current_segment[i + 1] - current_segment[i]

                # 最后一个梯度的计算方式为 0 减去最后一个数据点的值
                gradient[-1] = 0 - current_segment[-1]

                # 加速段平均加速度
                # 使用列表推导式提取正数
                positive_gradient = [x for x in gradient if x >= 0.5]
                try:
                    # 如果在try块中a_mean发生错误，那么a_mean就不会被初始化，可能导致后面使用a_mean报错，因为不存在
                    a_mean = sum(positive_gradient)/len(positive_gradient)
                except Exception as e:
                    print("发生错误:", e)
                    print("当前的 current_segment 数据:", current_segment)
                    # 如果你想查看完整的错误堆栈信息，可以使用 traceback
                    traceback.print_exc()

                a_meansegment.append(a_mean)

                # 最大加速度
                try:
                    a_max = max(positive_gradient)
                except Exception as e:
                    print("发生错误:", e)
                    print("当前的 current_segment 数据:", current_segment)
                a_maxsegment.append(a_max)

                # 减速段平均减速度
                negative_gradient = [x for x in gradient if x <= -0.5]
                try:
                    adown_mean = sum(negative_gradient) / len(negative_gradient)
                except Exception as e:
                    print("发生错误:", e)
                    print("当前的 current_segment 数据:", current_segment)
                adown_meansegment.append(adown_mean)

                # 匀速时间段
                cruise_gradient = [(current_segment,gradient) for current_segment,gradient in zip(current_segment,gradient)
                                    if -0.5 < abs(gradient) < 0.5 and current_segment > 1.8]

                # 怠速时间断
                idle_gradient = [(current_segment, gradient) for current_segment, gradient in zip(current_segment, gradient)
                                   if -0.5 < abs(gradient) < 0.5 and current_segment < 1.8]

                # 最大减速度
                adown_min = min(negative_gradient)
                adown_maxsegment.append(adown_min)

                # 怠速时间比
                idle_ratio = len(idle_gradient)/len(currentrun_segment)
                idle_ratiosegment.append(idle_ratio)

                # 加速时间比
                up_ratio = len(positive_gradient)/len(currentrun_segment)
                up_ratiosegment.append(up_ratio)

                # 减速时间比
                down_ratio = len(negative_gradient)/len(currentrun_segment)
                down_ratiosegment.append(down_ratio)

                # 匀速时间比
                cruise_ratio = len(cruise_gradient)/len(currentrun_segment)
                cruise_ratiosegment.append(cruise_ratio)

                # 速度标准差
                v_StdDev = np.std(current_segment)
                v_StdDevsegment.append(v_StdDev)

                # 加速度标准差
                p_StdDev = np.std(positive_gradient)
                p_StdDevsegment.append(p_StdDev)

                # 行驶里程
                S = sum(current_segment)/3.6
                s_segment.append(S)

                # 减速段减速度标准差
                n_StdDev = np.std(negative_gradient)
                n_StdDevsegment.append(n_StdDev)

                # 0-10km/h车速比例
                speed0_10 = [x for x in current_segment if 0 <= x <= 10]
                ratio0_10 = len(speed0_10)/len(current_segment)
                Ratio0_10.append(ratio0_10)

                # 10-20km/h车速比例
                speed10_20 = [x for x in current_segment if 10 <= x <= 20]
                ratio10_20 = len(speed10_20) / len(current_segment)
                Ratio10_20.append(ratio10_20)

                # 20-30km/h车速比例
                speed20_30 = [x for x in current_segment if 20 <= x <= 30]
                ratio20_30 = len(speed20_30) / len(current_segment)
                Ratio20_30.append(ratio20_30)

                # 30-40km/h车速比例
                speed30_40 = [x for x in current_segment if 30 <= x <= 40]
                ratio30_40 = len(speed30_40) / len(current_segment)
                Ratio30_40.append(ratio30_40)

                # 40-50km/h车速比例
                speed40_50 = [x for x in current_segment if 40 <= x <= 50]
                ratio40_50 = len(speed40_50) / len(current_segment)
                Ratio40_50.append(ratio40_50)

                # 50-60km/h车速比例
                speed50_60 = [x for x in current_segment if 50 <= x <= 60]
                ratio50_60 = len(speed50_60) / len(current_segment)
                Ratio50_60.append(ratio50_60)

                # 60-70km/h车速比例
                speed60_70 = [x for x in current_segment if 60 <= x <= 70]
                ratio60_70 = len(speed60_70) / len(current_segment)
                Ratio60_70.append(ratio60_70)

                # 70-80km/h车速比例
                speed70_80 = [x for x in current_segment if 70 <= x <= 80]
                ratio70_80 = len(speed70_80) / len(current_segment)
                Ratio70_80.append(ratio70_80)

                # 80-90km/h车速比例
                speed80_90 = [x for x in current_segment if 80 <= x <= 90]
                ratio80_90 = len(speed80_90) / len(current_segment)
                Ratio80_90.append(ratio80_90)

                # 90-100km/h车速比例
                speed90_100 = [x for x in current_segment if 90 <= x <= 100]
                ratio90_100 = len(speed90_100) / len(current_segment)
                Ratio90_100.append(ratio90_100)

                # 每100m的|a|>2的频率和每100m的|a|>10的频率
                # 设置初始值
                mileage = 0
                Mcount2 = 0
                Mcount10 = 0
                McountFrequency2 = []
                McountFrequency10 = []

                for i in range(len(current_segment)):   # 生成一个从0开始到列表长度减一的可迭代序列，因此这个循环可以把列表所有数据循环一遍，不多不少
                    mileage += current_segment[i]/3.6   # km/h转换为m/s
                    if abs(gradient[i]) > 2:
                        Mcount2 += 1
                    if abs(gradient[i]) > 10:
                        Mcount10 += 1
                    if mileage > 100:
                        McountFrequency2.append(Mcount2)
                        McountFrequency10.append(Mcount10)
                        mileage = 0
                        Mcount2 = 0
                        Mcount10 = 0

                # 每100m的|a|>2的频率平均值
                mmean100_2 = (sum(McountFrequency2) * 100) / int(S)   # 对于空序列sum返回0
                Mmean100_2.append(mmean100_2)

                # 每100m的|a|>2的频率最大值
                # if countFrequency2:   因为列表中有元素，视为true，如果是空列表[]才能被视为false
                if sum(McountFrequency2):
                    Mmax100_2.append(max(McountFrequency2))
                else:
                    Mmax100_2.append(0)

                # 每100m的|a|>10的频率平均值
                mmean100_10 = (sum(McountFrequency10) * 100) / int(S)
                Mmean100_10.append(mmean100_10)

                # 每100m的|a|>10的频率最大值
                if sum(McountFrequency10):
                    Mmax100_10.append(max(McountFrequency10))
                else:
                    Mmax100_10.append(0)

                # 加速度平方平均值
                n = len(gradient)
                if n == 0:
                    a2_mean.append(0)  # 避免除以零的情况
                else:
                    sum_of_squares = sum(x ** 2 for x in gradient)
                    a2_mean.append(sum_of_squares / n)

                # 每100s的|a|>2和|a|>10的频率
                Time = 0
                Tcount2 = 0
                Tcount10 = 0
                TcountFrequency2 = []
                TcountFrequency10 = []
                for i in range(len(current_segment)):  # 生成一个从0开始到列表长度减一的可迭代序列，因此这个循环可以把列表所有数据循环一遍，不多不少
                    Time += 1  # km/h转换为m/s
                    if abs(gradient[i]) > 2:
                        Tcount2 += 1
                    if abs(gradient[i]) > 10:
                        Tcount10 += 1
                    if Time > 100:
                        TcountFrequency2.append(Tcount2)
                        TcountFrequency10.append(Tcount10)
                        Time = 0
                        Tcount2 = 0
                        Tcount10 = 0

                # 每100s的|a|>2的频率平均值
                tmean100_2 = (sum(TcountFrequency2) * 100) / int(len(current_segment))  # 对于空序列sum返回0
                Tmean100_2.append(tmean100_2)

                # 每100s的|a|>2的频率最大值
                if sum(TcountFrequency2):
                    Tmax100_2.append(max(TcountFrequency2))
                else:
                    Tmax100_2.append(0)

                # 每100s的|a|>10的频率平均值
                tmean100_10 = (sum(TcountFrequency10) * 100) / int(len(current_segment))  # 对于空序列sum返回0
                Tmean100_10.append(tmean100_10)

                # 每100s的|a|>10的频率最大值
                if sum(TcountFrequency10):
                    Tmax100_10.append(max(TcountFrequency2))
                else:
                    Tmax100_10.append(0)

                # 运行时间
                segment_lengths.append(len(current_segment))
                current_segment = []
            # 如果当前段已经为空，说明连续两个0，忽略或根据需求处理
            else:
                current_segment.append(value)
        else:
            current_segment.append(value)

    # 检查最后一个段是否需要添加
    if sum(current_segment) != 0:
        if sum(1 for x in current_segment if x > 0) > 10:  # 没有使用列表推导式，使用生成器表达式，可以减少存储列表的内存占用
            current_segment.insert(0, 0)  # 由于判断时每一个数据段第一个0用于前面数据段的划分判断，因此需要再给数据段前面加0，但是第一个数据段就多了一个0
            segments.append(current_segment)

            # 平均速度
            v_mean = sum(current_segment) / len(current_segment)
            v_meansegment.append(v_mean)

            # 平均行驶速度
            # 使用 remove() 方法删除所有0元素
            currentrun_segment = list(current_segment)
            while 0 in currentrun_segment:
                currentrun_segment.remove(0)
            vr_mean = sum(currentrun_segment) / len(currentrun_segment)
            vr_meansegment.append(vr_mean)

            # 行驶时间
            T_run.append(len(currentrun_segment))

            # 最大速度
            v_max = max(current_segment)
            v_maxsegment.append(v_max)

            # 初始化一个与 data 长度相同的空列表，用于存储梯度
            gradient = [0.0] * len(current_segment)  # 初始化梯度列表，所有元素初始为 0.0

            # 计算前 n-1 个梯度
            for i in range(len(current_segment) - 1):
                gradient[i] = current_segment[i + 1] - current_segment[i]

            # 最后一个梯度的计算方式为 0 减去最后一个数据点的值
            gradient[-1] = 0 - current_segment[-1]

            # 加速段平均加速度
            # 使用列表推导式提取正数，0.15m/s转换为0.5km/h,0.5m/s转换为1.8km/h.
            positive_gradient = [x for x in gradient if x >= 0.5]
            try:
                a_mean = sum(positive_gradient) / len(positive_gradient)
            except Exception as e:
                print("发生错误:", e)
                print("当前的 current_segment 数据:", current_segment)
                # 如果你想查看完整的错误堆栈信息，可以使用 traceback
                traceback.print_exc()

            a_meansegment.append(a_mean)

            # 最大加速度
            try:
                a_max = max(positive_gradient)
            except Exception as e:
                print("发生错误:", e)
                print("当前的 current_segment 数据:", current_segment)
            a_maxsegment.append(a_max)

            # 平均减速度
            negative_gradient = [x for x in gradient if x <= -0.5]
            try:
                adown_mean = sum(negative_gradient) / len(negative_gradient)
            except Exception as e:
                print("发生错误:", e)
                print("当前的 current_segment 数据:", current_segment)
            adown_meansegment.append(adown_mean)

            # 匀速时间段
            cruise_gradient = [(current_segment, gradient) for current_segment, gradient in zip(current_segment, gradient)
                               if -0.5 < abs(gradient) < 0.5 and current_segment > 1.8]

            # 怠速时间段
            idle_gradient = [(current_segment, gradient) for current_segment, gradient in zip(current_segment, gradient)
                             if -0.5 < abs(gradient) < 0.5 and current_segment < 1.8]

            # 最大减速度
            adown_min = min(negative_gradient)
            adown_maxsegment.append(adown_min)

            # 怠速时间比
            idle_ratio = len(idle_gradient) / len(currentrun_segment)
            idle_ratiosegment.append(idle_ratio)

            # 加速时间比
            up_ratio = len(positive_gradient) / len(currentrun_segment)
            up_ratiosegment.append(up_ratio)

            # 减速时间比
            down_ratio = len(negative_gradient) / len(currentrun_segment)
            down_ratiosegment.append(down_ratio)

            # 匀速时间比
            cruise_ratio = len(cruise_gradient) / len(currentrun_segment)
            cruise_ratiosegment.append(cruise_ratio)

            # 速度标准差
            v_StdDev = np.std(current_segment)
            v_StdDevsegment.append(v_StdDev)

            # 加速度标准差
            p_StdDev = np.std(positive_gradient)
            p_StdDevsegment.append(p_StdDev)

            # 行驶里程
            S = sum(current_segment) / 3.6
            s_segment.append(S)

            # 减速度标准差
            n_StdDev = np.std(positive_gradient)
            n_StdDevsegment.append(n_StdDev)

            # 0-10km/h车速比例
            speed0_10 = [x for x in current_segment if 0 <= x < 10]
            ratio0_10 = len(speed0_10) / len(current_segment)
            Ratio0_10.append(ratio0_10)

            # 10-20km/h车速比例
            speed10_20 = [x for x in current_segment if 10 <= x < 20]
            ratio10_20 = len(speed10_20) / len(current_segment)
            Ratio10_20.append(ratio10_20)

            # 20-30km/h车速比例
            speed20_30 = [x for x in current_segment if 20 <= x < 30]
            ratio20_30 = len(speed20_30) / len(current_segment)
            Ratio20_30.append(ratio20_30)

            # 30-40km/h车速比例
            speed30_40 = [x for x in current_segment if 30 <= x < 40]
            ratio30_40 = len(speed30_40) / len(current_segment)
            Ratio30_40.append(ratio30_40)

            # 40-50km/h车速比例
            speed40_50 = [x for x in current_segment if 40 <= x < 50]
            ratio40_50 = len(speed40_50) / len(current_segment)
            Ratio40_50.append(ratio40_50)

            # 50-60km/h车速比例
            speed50_60 = [x for x in current_segment if 50 <= x < 60]
            ratio50_60 = len(speed50_60) / len(current_segment)
            Ratio50_60.append(ratio50_60)

            # 60-70km/h车速比例
            speed60_70 = [x for x in current_segment if 60 <= x < 70]
            ratio60_70 = len(speed60_70) / len(current_segment)
            Ratio60_70.append(ratio60_70)

            # 70-80km/h车速比例
            speed70_80 = [x for x in current_segment if 70 <= x < 80]
            ratio70_80 = len(speed70_80) / len(current_segment)
            Ratio70_80.append(ratio70_80)

            # 80-90km/h车速比例
            speed80_90 = [x for x in current_segment if 80 <= x < 90]
            ratio80_90 = len(speed80_90) / len(current_segment)
            Ratio80_90.append(ratio80_90)

            # 90-100km/h车速比例
            speed90_100 = [x for x in current_segment if 90 <= x <= 100]
            ratio90_100 = len(speed90_100) / len(current_segment)
            Ratio90_100.append(ratio90_100)

            # 每100m的|a|>2的频率和每100m的|a|>10的频率
            # 设置初始值
            mileage = 0
            Mcount2 = 0
            Mcount10 = 0
            McountFrequency2 = []
            McountFrequency10 = []

            for i in range(len(current_segment)):  # 生成一个从0开始到列表长度减一的可迭代序列，因此这个循环可以把列表所有数据循环一遍，不多不少
                mileage += current_segment[i] / 3.6  # km/h转换为m/s
                if abs(gradient[i]) > 2:
                    Mcount2 += 1
                if abs(gradient[i]) > 10:
                    Mcount10 += 1
                if mileage > 100:
                    McountFrequency2.append(Mcount2)
                    McountFrequency10.append(Mcount10)
                    mileage = 0
                    Mcount2 = 0
                    Mcount10 = 0

            # 每100m的|a|>2的频率平均值
            mmean100_2 = (sum(McountFrequency2) * 100) / int(S)  # 对于空序列sum返回0
            Mmean100_2.append(mmean100_2)

            # 每100m的|a|>2的频率最大值
            # if countFrequency2:   因为列表中有元素，视为true，如果是空列表[]才能被视为false
            if sum(McountFrequency2):
                Mmax100_2.append(max(McountFrequency2))
            else:
                Mmax100_2.append(0)

            # 每100m的|a|>10的频率平均值
            mmean100_10 = (sum(McountFrequency10) * 100) / int(S)
            Mmean100_10.append(mmean100_10)

            # 每100m的|a|>10的频率最大值
            if sum(McountFrequency10):
                Mmax100_10.append(max(McountFrequency10))
            else:
                Mmax100_10.append(0)

            # 加速度平方平均值
            n = len(gradient)
            if n == 0:
                a2_mean.append(0)  # 避免除以零的情况
            else:
                sum_of_squares = sum(x ** 2 for x in gradient)
                a2_mean.append(sum_of_squares / n)

            # 每100s的|a|>2和|a|>10的频率
            Time = 0
            Tcount2 = 0
            Tcount10 = 0
            TcountFrequency2 = []
            TcountFrequency10 = []
            for i in range(len(current_segment)):  # 生成一个从0开始到列表长度减一的可迭代序列，因此这个循环可以把列表所有数据循环一遍，不多不少
                Time += 1  # km/h转换为m/s
                if abs(gradient[i]) > 2:
                    Tcount2 += 1
                if abs(gradient[i]) > 10:
                    Tcount10 += 1
                if Time > 100:
                    TcountFrequency2.append(Tcount2)
                    TcountFrequency10.append(Tcount10)
                    Time = 0
                    Tcount2 = 0
                    Tcount10 = 0

            # 每100s的|a|>2的频率平均值
            tmean100_2 = (sum(TcountFrequency2) * 100) / int(len(current_segment))  # 对于空序列sum返回0
            Tmean100_2.append(tmean100_2)

            # 每100s的|a|>2的频率最大值
            if sum(TcountFrequency2):
                Tmax100_2.append(max(TcountFrequency2))
            else:
                Tmax100_2.append(0)

            # 每100s的|a|>10的频率平均值
            tmean100_10 = (sum(TcountFrequency10) * 100) / int(len(current_segment))  # 对于空序列sum返回0
            Tmean100_10.append(tmean100_10)

            # 每100s的|a|>10的频率最大值
            if sum(TcountFrequency10):
                Tmax100_10.append(max(TcountFrequency2))
            else:
                Tmax100_10.append(0)

            # 运行时间
            segment_lengths.append(len(current_segment))
            current_segment = []


    # 将列表转换为DataFrame
    data = {
    # '平均速度':v_meansegment,
    '平均行驶速度':vr_meansegment,
    '最大速度':v_maxsegment,
    '平均加速度':a_meansegment,
    '最大加速度':a_maxsegment,
    '平均减速度':adown_meansegment,
    '最大减速度':adown_maxsegment,
    '怠速时间占比':idle_ratiosegment,
    '加速时间占比':up_ratiosegment,
    '减速时间占比':down_ratiosegment,
    '匀速时间占比':cruise_ratiosegment,    #匀速行驶都可以视为巡航
    '速度标准差':v_StdDevsegment,
    '加速度标准差':p_StdDevsegment,
    # '行驶里程':s_segment,
    '运行时间':segment_lengths,
    '行驶时间':T_run,
    '减速度标准差':n_StdDevsegment,
    '0-10km/h的车速比例':Ratio0_10,
    '10-20km/h的车速比例':Ratio10_20,
    '20-30km/h的车速比例':Ratio20_30,
    '30-40km/h的车速比例':Ratio30_40,
    '40-50km/h的车速比例':Ratio40_50,
    '50-60km/h的车速比例':Ratio50_60,
    '60-70km/h的车速比例':Ratio60_70,
    '70-80km/h的车速比例':Ratio70_80,
    '80-90km/h的车速比例':Ratio80_90,
    '90-100km/h的车速比例':Ratio90_100,
    '每100m的a>2的频率平均值':Mmean100_2,
    '每100m的a>2的频率最大值':Mmax100_2,
    '每100m的a>10的频率平均值':Mmean100_10,
    '每100m的a>10的频率最大值':Mmax100_10,
    '每100s的a>2的频率平均值': Tmean100_2,
    '每100s的a>2的频率最大值': Tmax100_2,
    '每100s的a>10的频率平均值': Tmean100_10,
    '每100s的a>10的频率最大值': Tmax100_10,
    '加速度平方的均值':a2_mean
    }

    df = pd.DataFrame(data)

    # 保存到Excel文件
    df.to_excel(output_file, index=False, encoding='utf-8')

    print(f"数据已成功保存到 {output_file}")


# 示例使用
def main():
    # input_file = 'E:/recognition/实车数据/滤波处理1.xlsx'  # 输入Excel文件名，包含表头
    # output_file = 'E:/recognition/实车数据/片段划分1.xlsx'  # 输出Excel文件名
    input_file = 'E:/recognition/实车数据/滤波处理2.xlsx'  # 输入Excel文件名，包含表头
    output_file = 'E:/recognition/实车数据/片段划分2.xlsx'  # 输出Excel文件名

    # 如果你的数据列有列名，比如 'DataColumn'，可以这样调用：
    split_excel_data_with_header(input_file, output_file, column_name='Moving_Avg_VehSpd')

    # 如果没有列名，或者你想默认使用第一列，可以这样调用：
    # split_excel_data_with_header(input_file, output_file)

if __name__ == "__main__":
    main()