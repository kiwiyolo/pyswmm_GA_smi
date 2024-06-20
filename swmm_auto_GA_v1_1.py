# -*- coding: UTF-8 -*-
#
# Copyright (C) 2024 - 2024 QunLuo, Inc. All Rights Reserved 
#
# @Time    : 2024/6/9 15:36
# @Author  : QunLuo
# @Email   : 18098503978@163.com
# @File    : swmm_auto_GA
# @IDE     : PyCharm
# -----------------------------------------------------------------
import numpy as np
from swmm_api import read_out_file
import os
import math
import plotly.express as px
import pandas as pd
from pyswmm import Simulation
from tqdm import tqdm
from utils import nse
import re
from datetime import datetime
from scipy.signal import find_peaks



def load_observed_data(excel_path):
    """加载观测数据并处理时间索引"""
    df2 = pd.read_excel(excel_path, index_col=0)
    df2.index = df2.index.str.replace('24时', '0时')
    df2.index = pd.to_datetime(df2.index, format='%Y年%m月%d日%H时')
    # 转换索引为时间戳格式
    df2.index = pd.to_datetime(df2.index, format='%Y年%m月%d日%H时')
    df2.index = df2.index.strftime('%Y-%m-%d %H:%M:%S')
    return df2


def parse_dates(lines):
    """检索日期和时间"""
    date_pattern = re.compile(r'(\d{2}/\d{2}/\d{4})')
    time_pattern = re.compile(r'(\d{2}:\d{2}:\d{2})')

    report_start_date, report_start_time = None, None
    end_date, end_time = None, None

    for line in lines:
        if 'REPORT_START_DATE' in line:
            report_start_date = date_pattern.search(line).group(1)
        if 'REPORT_START_TIME' in line:
            report_start_time = time_pattern.search(line).group(1)
        if 'END_DATE' in line:
            end_date = date_pattern.search(line).group(1)
        if 'END_TIME' in line:
            end_time = time_pattern.search(line).group(1)

    return report_start_date, report_start_time, end_date, end_time


def format_datetime(date, time):
    """格式化日期和时间"""
    dt = datetime.strptime(f"{date} {time}", '%m/%d/%Y %H:%M:%S')
    return dt


def read_catch_inp(input_path):
    """读取inp文件中的report日期"""
    with open(input_path, 'r') as file:
        lines = file.readlines()

    report_start_date, report_start_time, end_date, end_time = parse_dates(lines)

    return format_datetime(report_start_date, report_start_time), format_datetime(end_date, end_time)


def read_modify_inp(N_imperv, N_perv, S_imperv, S_perv, Maxrate, Minrate, Decay, Drytime, Roughness,
                    idx, input_path, output_path,
                    new_report_start_datetime=None, new_end_datetime=None):
    """读取并修改输入文件"""
    with open(input_path, 'r') as file:
        lines = file.readlines()

    report_start_date, report_start_time, end_date, end_time = parse_dates(lines)

    if new_report_start_datetime:
        new_report_start_date, new_report_start_time = new_report_start_datetime.split()
    else:
        new_report_start_date, new_report_start_time = report_start_date, report_start_time

    if new_end_datetime:
        new_end_date, new_end_time = new_end_datetime.split()
    else:
        new_end_date, new_end_time = end_date, end_time

    substartindex = next(i for i, line in enumerate(lines) if '[SUBAREAS]' in line)
    subendindex = next(i for i, line in enumerate(lines) if '[INFILTRATION]' in line)
    infilendindex = next(i for i, line in enumerate(lines) if '[JUNCTIONS]' in line)
    CONDUITSstartindex = next(i for i, line in enumerate(lines) if '[CONDUITS]' in line)
    CONDUITSendindex = next(i for i, line in enumerate(lines) if '[XSECTIONS]' in line)

    for i in range(substartindex+3, subendindex):
        parts = lines[i].split()
        if parts:
            parts[1:5] = [N_imperv[idx], N_perv[idx], S_imperv[idx], S_perv[idx]]
            lines[i] = '      '.join(map(str, parts)) + '\n'

    for i in range(subendindex+3, infilendindex):
        parts = lines[i].split()
        if parts:
            parts[1:5] = [Maxrate[idx], Minrate[idx], Decay[idx], Drytime[idx]]
            lines[i] = '      '.join(map(str, parts)) + '\n'

    for i in range(CONDUITSstartindex+3, CONDUITSendindex):
        parts = lines[i].split()
        if parts:
            parts[4] = Roughness[idx]
            lines[i] = '      '.join(map(str, parts)) + '\n'

    for i in range(len(lines)):
        if 'REPORT_START_DATE' in lines[i]:
            lines[i] = f"REPORT_START_DATE    {new_report_start_date}\n"
        if 'REPORT_START_TIME' in lines[i]:
            lines[i] = f"REPORT_START_TIME    {new_report_start_time}\n"
        if 'END_DATE' in lines[i]:
            lines[i] = f"END_DATE             {new_end_date}\n"
        if 'END_TIME' in lines[i]:
            lines[i] = f"END_TIME             {new_end_time}\n"

    with open(output_path, 'w') as file:
        file.writelines(lines)
    return format_datetime(new_report_start_date, new_report_start_time), format_datetime(new_end_date, new_end_time)


def run_simulation(simulation_input_path):
    """运行SWMM模型"""
    with Simulation(simulation_input_path) as sim:
        sim.execute()


def calculate_relative_errors_(Qs, Qm):
    # Find peaks for Qs and Qm
    peaks_Qs, _ = find_peaks(Qs)
    peaks_Qm, _ = find_peaks(Qm)
    if len(peaks_Qm) == len(peaks_Qs):
        # Get peak values and positions
        peak_values_Qs = Qs.iloc[peaks_Qs]
        peak_positions_Qs = peaks_Qs

        peak_values_Qm = Qm.iloc[peaks_Qm]
        peak_positions_Qm = peaks_Qm

        # Calculate relative errors for peak values
        peak_values_error = np.exp(-np.abs((peak_values_Qs.values - peak_values_Qm.values)))

        # Calculate relative errors for peak positions
        peak_positions_error = np.exp(-np.abs((peak_positions_Qs - peak_positions_Qm)))
    else:
        peak_values_error = peak_positions_error = 0
    return float(peak_values_error), float(peak_positions_error)

def calculate_relative_errors_(Qs, Qm):
    # Find peaks for Qs and Qm
    Qsm = Qs.max()
    Qmm = Qm.max()
    peaks_Qs, _ = find_peaks(Qs, prominence=0.5*Qsm)
    peaks_Qm, _ = find_peaks(Qm, prominence=0.5*Qmm)
    if len(peaks_Qm) == len(peaks_Qs):
        # Get peak values and positions
        peak_values_Qs = Qs.iloc[peaks_Qs]
        peak_positions_Qs = peaks_Qs

        peak_values_Qm = Qm.iloc[peaks_Qm]
        peak_positions_Qm = peaks_Qm

        # Calculate relative errors for peak values
        peak_values_error = np.exp(-np.abs((peak_values_Qs.values - peak_values_Qm.values)))

        # Calculate relative errors for peak positions
        peak_positions_error = np.exp(-np.abs((peak_positions_Qs - peak_positions_Qm)))
    else:
        peak_values_error = peak_positions_error = 0
    return float(peak_values_error), float(peak_positions_error)

def calculate_relative_errors(Qs, Qm):
    # Find peaks for Qs and Qm
    Qsm = Qs.max()
    Qmm = Qm.max()
    peaks_Qs, _ = find_peaks(Qs, prominence=0.5*Qsm)
    peaks_Qm, _ = find_peaks(Qm, prominence=0.5*Qmm)
    if len(peaks_Qm) == len(peaks_Qs):
        # Get peak values and positions
        peak_values_Qs = Qs.iloc[peaks_Qs]
        peak_positions_Qs = peaks_Qs

        peak_values_Qm = Qm.iloc[peaks_Qm]
        peak_positions_Qm = peaks_Qm

        # Calculate relative errors for peak values
        peak_values_error = np.exp(-np.abs((peak_values_Qs.values - peak_values_Qm.values)))

        # Calculate relative errors for peak positions
        peak_positions_error = np.exp(-np.abs((peak_positions_Qs - peak_positions_Qm)))
    else:
        peak_values_error = peak_positions_error = 0
    return float(peak_values_error), float(peak_positions_error)


def calculate_fitness(N_imperv, N_perv, S_imperv, S_perv, Maxrate, Minrate, Decay, Drytime, Roughness,
                      input_file_path, modified_input_path, output_file_path,
                      observed_data, pop_size):
    """计算适应度函数"""
    results = []
    for i in range(pop_size):
        # report_start, report_end = read_modify_inp(N_imperv, S_imperv, Minrate, Roughness,
        #                                            i, input_file_path, modified_input_path)
        total_nse = 0
        count = 0
        for j in range(len(input_file_path)):
            report_start, report_end = read_modify_inp(N_imperv, N_perv, S_imperv, S_perv, Maxrate, Minrate, Decay,
                                                       Drytime, Roughness, i, input_file_path[j], modified_input_path[j])
            run_simulation(modified_input_path[j])

            out = read_out_file(output_file_path[j])
            df = out.to_frame()
            node = df['node']
            node.columns = ['_'.join(col) for col in node.columns.values]
            Qm = node['Y035677_total_inflow']

            # 确保 df 的索引是 DatetimeIndex 类型
            if not isinstance(observed_data.index, pd.DatetimeIndex):
                observed_data.index = pd.to_datetime(observed_data.index)
            # 使用 loc 方法提取指定范围的行
            # 将 report_start 增加 1 小时
            report_start += pd.Timedelta(hours=1)
            filtered_obs = observed_data.loc[report_start:report_end]
            Qs = filtered_obs['平均值']
            # 计算峰值及其位置
            # peak_values_error, peak_positions_error = calculate_relative_errors(Qs, Qm)
            # Ensure nse, peak_values_error, and peak_positions_error are floats
            # eval = (float(nse(Qs, Qm)) + peak_values_error + peak_positions_error) / 3.0
            total_nse += float(nse(Qs, Qm))
            count += 1
        eval = total_nse / count if count > 0 else 0
        results.append(eval)

    return results
def calculate_fitness_(N_imperv, N_perv, S_imperv, S_perv, Maxrate, Minrate, Decay, Drytime, Roughness, input_file_path,
                        modified_input_path, output_file_path,
                      observed_data, pop_size):
    """计算适应度函数"""
    results = []
    for i in range(pop_size):
        report_start, report_end = read_modify_inp(N_imperv, N_perv, S_imperv, S_perv, Maxrate, Minrate, Decay,
                                                   Drytime, Roughness, i, input_file_path, modified_input_path)
        run_simulation(modified_input_path)

        out = read_out_file(output_file_path)
        df = out.to_frame()
        node = df['node']
        node.columns = ['_'.join(col) for col in node.columns.values]
        Qm = node['Y035677_total_inflow']

        # 确保 df 的索引是 DatetimeIndex 类型
        if not isinstance(observed_data.index, pd.DatetimeIndex):
            observed_data.index = pd.to_datetime(observed_data.index)
        # 使用 loc 方法提取指定范围的行
        # 将 report_start 增加 1 小时
        report_start += pd.Timedelta(hours=1)
        filtered_obs = observed_data.loc[report_start:report_end]
        Qs = filtered_obs['平均值']
        # 计算峰值及其位置
        peak_values_error, peak_positions_error = calculate_relative_errors(Qs, Qm)

        eval = (nse(Qs, Qm) + peak_values_error + peak_positions_error) / 3
        results.append(eval)

    return results


def translate_DNA(pop, param_bounds, dna_size):
    """将二进制DNA转换为实际参数值"""
    params = {}
    param_names = ['N_imperv', 'N_perv', 'S_imperv', 'S_perv', 'Maxrate', 'Minrate', 'Decay', 'Drytime', 'Roughness']
    for i, name in enumerate(param_names):
        pop_section = pop[:, i * dna_size:(i + 1) * dna_size]
        params[name] = pop_section.dot(2 ** np.arange(dna_size)[::-1]) / (2 ** dna_size - 1) * (
                    param_bounds[name][1] - param_bounds[name][0]) + param_bounds[name][0]
    return [params[name] for name in param_names]


def get_fitness(pop, input_file_paths,
                        modified_input_paths, output_file_paths, observed_data, param_bounds, dna_size, pop_size):
    """获取种群的适应度"""
    params = translate_DNA(pop, param_bounds, dna_size)
    fitness = calculate_fitness(*params, input_file_paths,
                        modified_input_paths, output_file_paths, observed_data, pop_size)
    param_names = ['N_imperv', 'N_perv', 'S_imperv', 'S_perv', 'Maxrate', 'Minrate', 'Decay', 'Drytime', 'Roughness']
    with open('./kw_03_saved_models/shiyingdu.txt', 'a') as file:
        file.write(f"{','.join(map(str, fitness))}\n")
        max_idx = np.argmax(fitness)
        # Write parameter values to file
        param_values = {name: params[i][max_idx] for i, name in enumerate(param_names)}
        file.write("参数值： " + str(param_values) + "\n")
        # file.write(f"{max(fitness)}\n")
    return np.array(fitness)


def crossover_and_mutation(pop, dna_size, pop_size, crossover_rate=0.8, mutation_rate=0.1):
    """交叉和变异操作"""
    new_pop = []
    for father in pop:
        child = father.copy()
        if np.random.rand() < crossover_rate:
            mother = pop[np.random.randint(pop_size)]
            cross_points = np.random.randint(0, dna_size * 9)
            child[cross_points:] = mother[cross_points:]
        if np.random.rand() < mutation_rate:
            mutate_point = np.random.randint(0, dna_size * 9)
            child[mutate_point] ^= 1
        new_pop.append(child)
    return np.array(new_pop)


def select(pop, fitness, pop_size):
    """选择操作"""
    idx = np.random.choice(np.arange(pop_size), size=pop_size, replace=True, p=fitness / fitness.sum())
    return pop[idx]


def print_info(pop, output_file_paths, observed_data, param_bounds, dna_size, pop_size):
    """打印信息"""
    fitness = get_fitness(pop, input_file_paths,
                          modified_input_paths, output_file_paths, observed_data, param_bounds, dna_size, pop_size)
    max_idx = np.argmax(fitness)
    print(f"max_fitness: {fitness[max_idx]}")
    params = translate_DNA(pop, param_bounds, dna_size)
    param_names = ['N_imperv', 'N_perv', 'S_imperv', 'S_perv', 'Maxrate', 'Minrate', 'Decay', 'Drytime', 'Roughness']
    print("最优的基因型：", pop[max_idx])
    print("参数值：", {name: params[i][max_idx] for i, name in enumerate(param_names)})


if __name__ == "__main__":
    # 文件路径
    excel_path = './kw_01_data/PSK_if.xlsx'
    input_file_paths = ['./kw_03_saved_models/race_model.inp', './kw_03_saved_models/race_model_rain2.inp']
    modified_input_paths = ['./kw_03_saved_models/race_model_mod.inp', './kw_03_saved_models/race_model_rain2_mod.inp']
    output_file_paths = ['./kw_03_saved_models/race_model_mod.out', './kw_03_saved_models/race_model_rain2_mod.out']

    # 加载观测数据
    observed_data = load_observed_data(excel_path)

    # 遗传算法常量
    DNA_SIZE = 10   # 构建超参数所代表的DNA的二进制位数，表征交叉编译的复杂度
    POP_SIZE = 100   # 构建超参数赋值列表长度（单次组合递增密度）
    CROSSOVER_RATE = 0.8
    MUTATION_RATE = 0.1
    N_GENERATIONS = 100   # 迭代筛选出最优的组合（迭代更新组合次数）

    # # 参数的边界
    # PARAM_BOUNDS = {
    #     'N_imperv': [1.1016617790811338, 1.1016617790811338],   # [0, 1]  [0.1, 0.1]
    #     'N_perv': [0.6940371456500489, 0.6940371456500489],   # [0, 1]    [0.1, 0.1]
    #     'S_imperv': [7.126099706744868, 7.126099706744868],   # [0, 5]    [0.01, 0.01]  增加会使得小降水事件被平滑，径流推迟
    #     'S_perv': [0.6011730205278593, 0.6011730205278593],   # [0, 5]  [0.01, 0.01]  几乎没什么影响，说明基本没有透水区
    #     'Maxrate': [73.16715542521993, 73.16715542521993],   # [5, 100]   [50, 50]  几乎没什么影响
    #     'Minrate': [5.1759530791788855, 5.1759530791788855],   # [0, 5]   [1, 1]    增大会使峰值变异性增加，瘦长
    #     'Decay': [11.099706744868037, 11.099706744868037],   # [0, 20]    [1, 1]    对峰值有轻微正向影响
    #     'Drytime': [3.725317693059629, 3.725317693059629],   # [0, 10]   [1, 1]  几乎没什么影响
    #     'Roughness': [0.027126099706744868, 0.027126099706744868]
    # }

    #  {'N_imperv': 1.2792766373411537, 'N_perv': 0.7145650048875856, 'S_imperv': 8.064516129032258,
    #  'S_perv': 1.1094819159335287, 'Maxrate': 67.0772238514174, 'Minrate': 0.6842619745845552,
    #  'Decay': 14.52590420332356, 'Drytime': 3.8367546432062563, 'Roughness': 0.01348973607038123}
    #
    # 参数的边界
    PARAM_BOUNDS = {
        'N_imperv': [1, 3],   # [0, 1]  [0.1, 0.1]
        'N_perv': [0, 1],   # [0, 1]    [0.1, 0.1]
        'S_imperv': [5, 30],   # [0, 5]    [0.01, 0.01]
        'S_perv': [0, 5],   # [0, 5]  [0.01, 0.01]
        'Maxrate': [50, 100],   # [5, 100]   [50, 50]
        'Minrate': [5, 20],   # [0, 5]   [1, 1]
        'Decay': [5, 20],   # [0, 20]    [1, 1]
        'Drytime':  [3, 5],    # [0, 10]   [1, 1]
        'Roughness': [0, 0.05]
    }

    # 初始化种群
    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE * 9))
    fitness_history = []

    for gen in tqdm(range(N_GENERATIONS), desc="Genetic Algorithm Progress"):
        print(f'第{gen + 1}次迭代')
        pop = crossover_and_mutation(pop, DNA_SIZE, POP_SIZE, CROSSOVER_RATE, MUTATION_RATE)
        fitness = get_fitness(pop, input_file_paths,
                              modified_input_paths, output_file_paths, observed_data, PARAM_BOUNDS, DNA_SIZE, POP_SIZE)
        fitness_history.append(fitness.max())
        pop = select(pop, fitness, POP_SIZE)

    print_info(pop, output_file_paths, observed_data, PARAM_BOUNDS, DNA_SIZE, POP_SIZE)

    for i, name in enumerate(output_file_paths):
        # grab figure data
        out = read_out_file(output_file_paths[i])
        report_start, report_end = read_catch_inp(input_file_paths[i])
        # run_simulation(modified_input_path)
        # 将 report_start 增加 1 小时
        report_start += pd.Timedelta(hours=1)
        filtered_obs = observed_data.loc[report_start:report_end]
        df = out.to_frame()
        node = df['node']
        node.columns = ['_'.join(col) for col in node.columns.values]
        Qm = node['Y035677_total_inflow']
        Qs = filtered_obs['平均值']
        df3 = pd.concat([Qm, Qs], axis=1)
        # plot line figure
        fig = px.line(df3, x=df3.index, y=['平均值', 'Y035677_total_inflow'], labels=['观测流量(m3/s)', '模拟流量(m3/s)'])
        fig.show()
