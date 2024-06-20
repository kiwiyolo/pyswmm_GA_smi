# -*- coding: UTF-8 -*-
#
# Copyright (C) 2024 - 2024 QunLuo, Inc. All Rights Reserved 
#
# @Time    : 2024/4/18 16:37
# @Author  : QunLuo
# @Email   : 18098503978@163.com
# @File    : Simulator
# @IDE     : PyCharm
# -----------------------------------------------------------------
from pyswmm import Simulation
import os


# 获取当前脚本文件所在的路径
current_path = os.path.dirname(os.path.abspath(__file__))
# 构建上一级文件夹的路径
parent_path = os.path.dirname(current_path)

inp_root = parent_path + '/kw_03_saved_models/tzswmm-bbweir.inp'
sim = Simulation(inp_root)
sim.execute()

import swmm5


def run_swmm(input_file):
    """
    Run SWMM model with given input file.
    """
    swmm_model = swmm5.swmm_run(input_file, reportfile=None, outputfile=None)
    return swmm_model


def modify_input(input_file, parameter_changes):
    """
    Modify input file with specified parameter changes.
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Apply parameter changes
    for param, value in parameter_changes.items():
        for i, line in enumerate(lines):
            if param in line:
                lines[i] = line.replace(param, str(value))

    # Write modified input file
    with open('modified_input.inp', 'w') as f:
        f.writelines(lines)


def calibrate_model(input_file, parameter_changes):
    """
    Calibrate SWMM model with specified parameter changes.
    """
    # Modify input file
    modify_input(input_file, parameter_changes)

    # Run SWMM model with modified input file
    swmm_model = run_swmm('modified_input.inp')

    # Add calibration logic here, e.g., compare model output with observed data
    # and adjust parameters accordingly


if __name__ == "__main__":
    input_file = 'original_input.inp'
    parameter_changes = {'RAINGAGE': 'new_raingage', 'HYDROGRAPH': 'new_hydrograph'}
    calibrate_model(input_file, parameter_changes)