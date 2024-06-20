# -*- coding: UTF-8 -*-
#
# Copyright (C) 2024 - 2024 QunLuo, Inc. All Rights Reserved 
#
# @Time    : 2024/4/19 17:00
# @Author  : QunLuo
# @Email   : 18098503978@163.com
# @File    : simulator_p
# @IDE     : PyCharm
# -----------------------------------------------------------------
def modify_rainfall_data(input_file, new_rainfall_data):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # 查找降水时序数据的起始行和结束行
    start_line = None
    end_line = None
    for i, line in enumerate(lines):
        if line.strip().startswith('[TIMESERIES]'):
            start_line = i
        elif line.strip().startswith('[END]') and start_line is not None:
            end_line = i
            break

    # 如果找到了降水时序数据的起始行和结束行，则替换其中的数据
    if start_line is not None and end_line is not None:
        lines[start_line+2:end_line] = new_rainfall_data

    # 将修改后的内容写回输入文件
    with open(input_file, 'w') as f:
        f.writelines(lines)

if __name__ == "__main__":
    input_file = "your_input_file.inp"  # 替换为你的SWMM输入文件路径
    new_rainfall_data = [
        "10/01/2024 00:00, 0.1\n",  # 示例降水时序数据，格式为日期时间, 降水量
        "10/01/2024 01:00, 0.2\n",
        # 添加更多的降水数据...
    ]
    modify_rainfall_data(input_file, new_rainfall_data)