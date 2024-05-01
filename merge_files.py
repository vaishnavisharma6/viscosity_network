import numpy as np
import random
import os
import argparse
from os.path import join
import shutil
import time, glob

home_dir = os.path.expanduser('~')
current_dir = os.getcwd()
main_dir = join(home_dir, "SSFR")
linear_dir = join(main_dir, "linear_training")
burg_dir = join(main_dir, "burg_training")

linear = []
for i in range (2, 3):
    linear_path1_dir = join(linear_dir, "output_model2_{}_40".format(i))
    linear_path2_dir = join(linear_dir, "output_model2_{}_80".format(i))
    linear_path3_dir = join(linear_dir, "output_model2_{}_120".format(i))

    linear_path1 = join(linear_path1_dir, "nodal_vs_alpha")
    linear_path2 = join(linear_path2_dir, "nodal_vs_alpha")
    linear_path3 = join(linear_path3_dir, "nodal_vs_alpha")

    for file in os.listdir(linear_path1):
        if file.endswith(".txt"):
            linear.append(join(linear_path1, file))

    for file in os.listdir(linear_path2):
        if file.endswith(".txt"):
            linear.append(join(linear_path2, file))

    for file in os.listdir(linear_path3):
        if file.endswith(".txt"):
            linear.append(join(linear_path3, file))

with open('linear.txt', 'w') as file:
    for name in linear:
            with open(name, encoding="utf-8", errors = 'ignore') as infile:
                file.write(infile.read())  

d = np.loadtxt("linear.txt")
m = d.shape[0]
n = d.shape[1]
print(m)
print(n)

inputs = []

# burgers' equation files merge

for i in range(1,4):

    burg_path1_dir = join(burg_dir, "output_model2_{}_40_domain1".format(i))
    burg_path2_dir = join(burg_dir, "output_model2_{}_80_domain1".format(i))
    burg_path3_dir = join(burg_dir, "output_model2_{}_120_domain1".format(i))

    burg_path1 = join(burg_path1_dir, "nodal_vs_alpha")
    burg_path2 = join(burg_path2_dir, "nodal_vs_alpha")
    burg_path3 = join(burg_path3_dir, "nodal_vs_alpha")

    for file in os.listdir(burg_path1):
        if file.endswith(".txt"):
            inputs.append(join(burg_path1, file))

    for file in os.listdir(burg_path2):
        if file.endswith(".txt"):
            inputs.append(join(burg_path2, file))

    for file in os.listdir(burg_path3):
        if file.endswith(".txt"):
            inputs.append(join(burg_path3, file))


for i in range(14,15):

    burg_path1_dir = join(burg_dir, "output_model2_{}_40_domain1".format(i))
    burg_path2_dir = join(burg_dir, "output_model2_{}_80_domain1".format(i))
    burg_path3_dir = join(burg_dir, "output_model2_{}_120_domain1".format(i))

    burg_path1 = join(burg_path1_dir, "nodal_vs_alpha")
    burg_path2 = join(burg_path2_dir, "nodal_vs_alpha")
    burg_path3 = join(burg_path3_dir, "nodal_vs_alpha")

    for file in os.listdir(burg_path1):
        if file.endswith(".txt"):
            inputs.append(join(burg_path1, file))

    for file in os.listdir(burg_path2):
        if file.endswith(".txt"):
            inputs.append(join(burg_path2, file))

    for file in os.listdir(burg_path3):
        if file.endswith(".txt"):
            inputs.append(join(burg_path3, file))            

for i in range(2,3):
    
    burg_path1_d2 = join(burg_dir, "output_model2_{}_40_domain2".format(i))
    burg_path2_d2 = join(burg_dir, "output_model2_{}_80_domain2".format(i))
    burg_path3_d2 = join(burg_dir, "output_model2_{}_120_domain2".format(i))
    burg_path4_d2 = join(burg_dir, "output_model2_{}_200_domain2".format(i))

    burg1_d2 = join(burg_path1_d2, "nodal_vs_alpha")
    burg2_d2 = join(burg_path2_d2, "nodal_vs_alpha")
    burg3_d2 = join(burg_path3_d2, "nodal_vs_alpha")
    burg4_d2 = join(burg_path4_d2, "nodal_vs_alpha")



    for file in os.listdir(burg1_d2):
        if file.endswith(".txt"):
            inputs.append(join(burg1_d2, file))

    for file in os.listdir(burg2_d2):
        if file.endswith(".txt"):
            inputs.append(join(burg2_d2, file))

    for file in os.listdir(burg3_d2):
        if file.endswith(".txt"):
            inputs.append(join(burg3_d2, file))

    for file in os.listdir(burg4_d2):
        if file.endswith(".txt"):
            inputs.append(join(burg4_d2, file))

with open('burgers.txt', 'w') as output:
    for name in inputs:
        with open(name, encoding="utf-8", errors = 'ignore') as infile:
            output.write(infile.read())                                                        
d = np.loadtxt('burgers.txt')
m = d.shape[0]
n = d.shape[1]
print(m)
print(n)
    
linear_data = join(current_dir, "linear.txt")
burgers_data = join(current_dir, "burgers.txt")

with open(linear_data) as fp:
    data = fp.read()

with open(burgers_data) as fp1:
    # data = fp1.read()
    data += "\n"
    data1 = fp1.read()
    data += data1

with open('final_data.txt', 'w') as file:
    file.write(data)

d = np.loadtxt("final_data.txt")
m = d.shape[0]
n = d.shape[1]
print(m)
print(n)

np.random.shuffle(d)
np.savetxt("data.txt", d)

# remove any duplicate rows

rows_seen = set()
with open('all_data.txt', 'w') as file1:
    for each_line in open('data.txt', 'r'):
        if each_line not in rows_seen:
            file1.write(each_line)
            rows_seen.add(each_line)

d1 = np.loadtxt("all_data.txt")
m = d1.shape[0]
n = d1.shape[1]

print(m)
print(n)




