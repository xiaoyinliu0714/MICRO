import random
import pprint
import time
import uuid
import tempfile
import os
import re
from copy import copy
from socket import gethostname
import pickle

import numpy as np

import absl.flags
from absl import logging
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags
from ml_collections.config_dict import config_dict

import torch

# generate xml assets path: gym_xml_path
def generate_xml_path():
    import gym, os
    xml_path = os.path.join(gym.__file__[:-11], 'envs/mujoco/assets')

    assert os.path.exists(xml_path)
    print("gym_xml_path: ",xml_path)

    return xml_path

gym_xml_path = generate_xml_path()


def record_data(file, content):
    with open(file, 'a+') as f:
        f.write('{}\n'.format(content))

def check_path(path):
    try:
        if not os.path.exists(path):
            os.mkdir(path)
    except FileExistsError:
        pass

    return path


def update_xml(index, env_name):
    xml_name = parse_xml_name(env_name)
    os.system('cp ../MICRO/robust/xml_path/{0}/{1} {2}/{1}}'.format(index, xml_name, gym_xml_path))

    time.sleep(0.2)


def parse_xml_name(env_name):
    if 'walker' in env_name.lower():
        xml_name = "walker2d.xml"
    elif 'hopper' in env_name.lower():
        xml_name = "hopper.xml"
    elif 'halfcheetah' in env_name.lower():
        xml_name = "half_cheetah.xml"
    elif "ant" in env_name.lower():
        xml_name = "ant.xml"
    else:
        raise RuntimeError("No available environment named \'%s\'" % env_name)

    return xml_name


def update_source_env(env_name):
    xml_name = parse_xml_name(env_name)

    os.system(
        'cp ../MICRO/robust/xml_path/real_file/{0} {1}/{0}'.format(xml_name, gym_xml_path))

    time.sleep(0.2)


# gravity
def update_target_env_gravity(variety_degree, env_name):
    old_xml_name = parse_xml_name(env_name)
    # create new xml 
    xml_name = "{}_gravityx{}.xml".format(old_xml_name.split(".")[0], variety_degree)

    with open('../MICRO/robust/xml_path/real_file/{}'.format(old_xml_name), "r+") as f:

        new_f = open('../MICRO/robust/xml_path/sim_file/{}'.format(xml_name), "w+")
        for line in f.readlines():
            if "gravity" in line:
                pattern = re.compile(r"gravity=\"(.*?)\"")
                a = pattern.findall(line)
                gravity_list = a[0].split(" ")
                new_gravity_list = []
                for num in gravity_list:
                    new_gravity_list.append(variety_degree * float(num))

                replace_num = " ".join(str(i) for i in new_gravity_list)
                replace_num = "gravity=\"" + replace_num + "\""
                sub_result = re.sub(pattern, str(replace_num), line)

                new_f.write(sub_result)
            else:
                new_f.write(line)

        new_f.close()
    # replace the default gym env with newly-revised env
    os.system(
        'cp ../MICRO/robust/xml_path/sim_file/{0} {1}/{2}'.format(xml_name, gym_xml_path, old_xml_name))#保存路径
    time.sleep(0.2)

# friction
def update_target_env_friction(variety_degree, env_name):
    old_xml_name = parse_xml_name(env_name)
    # create new xml 
    xml_name = "{}_frictionx{}.xml".format(old_xml_name.split(".")[0], variety_degree)

    with open('../MICRO/robust/xml_path/real_file/{}'.format(old_xml_name), "r+") as f:

        new_f = open('../MICRO/robust/xml_path/sim_file/{}'.format(xml_name), "w")
        for line in f.readlines():
            if "friction" in line:
                pattern = re.compile(r"friction=\"(.*?)\"")
                a = pattern.findall(line)
                friction_list = a[0].split(" ")
                new_friction_list = []
                for num in friction_list:
                    new_friction_list.append(variety_degree * float(num))

                replace_num = " ".join(str(i) for i in new_friction_list)
                replace_num = "friction=\"" + replace_num + "\""
                sub_result = re.sub(pattern, str(replace_num), line)

                new_f.write(sub_result)
            else:
                new_f.write(line)

        new_f.close()

    # replace the default gym env with newly-revised env
    os.system(
        'cp ../MICRO/robust/xml_path/sim_file/{0} {1}/{2}'.format(xml_name, gym_xml_path, old_xml_name))

    time.sleep(0.2)

# friction and gravity
def update_target_env(variety_degree_friction, variety_degree_gravity, env_name):
    old_xml_name = parse_xml_name(env_name)
    # create new xml
    xml_name = "{}_target.xml".format(old_xml_name.split(".")[0])

    with open('../MICRO/robust/xml_path/real_file/{}'.format(old_xml_name),
              "r+") as f:
        new_f = open('../MICRO/robust/xml_path/sim_file/{}'.format(xml_name),
                     "w")
        for line in f.readlines():

            if "friction" in line and "gravity" in line:
                pattern = re.compile(r"friction=\"(.*?)\"")
                a = pattern.findall(line)
                friction_list = a[0].split(" ")
                new_friction_list = []
                for num in friction_list:
                    new_friction_list.append(variety_degree_friction * float(num))

                replace_num = " ".join(str(i) for i in new_friction_list)
                replace_num = "friction=\"" + replace_num + "\""
                sub_result = re.sub(pattern, str(replace_num), line)

                pattern_1 = re.compile(r"gravity=\"(.*?)\"")
                a = pattern_1.findall(sub_result)
                gravity_list = a[0].split(" ")
                new_gravity_list = []
                for num in gravity_list:
                    new_gravity_list.append(variety_degree_gravity * float(num))

                replace_num = " ".join(str(i) for i in new_gravity_list)
                replace_num = "gravity=\"" + replace_num + "\""
                sub_result_1 = re.sub(pattern_1, str(replace_num), sub_result)
                new_f.write(sub_result_1)
            elif "friction" in line and "gravity" not in line:
                pattern = re.compile(r"friction=\"(.*?)\"")
                a = pattern.findall(line)
                friction_list = a[0].split(" ")
                new_friction_list = []
                for num in friction_list:
                    new_friction_list.append(variety_degree_friction * float(num))

                replace_num = " ".join(str(i) for i in new_friction_list)
                replace_num = "friction=\"" + replace_num + "\""
                sub_result = re.sub(pattern, str(replace_num), line)
                new_f.write(sub_result)

            elif "friction" not in line and "gravity" in line:
                pattern = re.compile(r"gravity=\"(.*?)\"")
                a = pattern.findall(line)
                gravity_list = a[0].split(" ")
                new_gravity_list = []
                for num in gravity_list:
                    new_gravity_list.append(variety_degree_gravity * float(num))

                replace_num = " ".join(str(i) for i in new_gravity_list)
                replace_num = "gravity=\"" + replace_num + "\""
                sub_result = re.sub(pattern, str(replace_num), line)

                new_f.write(sub_result)
            else:
                new_f.write(line)

        new_f.close()

    # replace the default gym env with newly-revised env
    os.system(
        'cp ../MICRO/robust/xml_path/sim_file/{0} {1}/{2}'.format(xml_name,gym_xml_path,old_xml_name))

    time.sleep(0.2)