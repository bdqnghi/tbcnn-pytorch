import shutil
import os
import random
from shutil import copyfile
from concurrent.futures import ThreadPoolExecutor

# ROOT = "/home/nghibui/codes/bi-tbcnn/"
src_dir = "ProgramData_raw"
tgt_dir = "ProgramData_pb"

algo_directories = os.listdir(src_dir)

# langs = ["cpp","java"]

def execute_command(src_path,tgt_path):
    cmd = "docker run -v $(pwd):/e -it yijun/fast:built -p " + src_path + " " + tgt_path
    print(cmd)
    os.system(cmd)

with ThreadPoolExecutor(max_workers=100) as executor:        
    for i, algo in enumerate(algo_directories):
       

        algo_directory = os.path.join(src_dir,algo)

       
        algo_directory_splits = algo_directory.split("/")

        new_path = os.path.join(src_dir,algo )

        if not os.path.exists(new_path):
            os.makedirs(new_path)

        # for lang in langs:

        # algo_directory_lang = os.path.join(algo_directory,lang)
        files = os.listdir(algo_directory)

        for file in files:
            file_path = os.path.join(algo_directory, file)
             
            pb_new_path = os.path.join(tgt_dir,  algo)

            if not os.path.exists(pb_new_path):
                os.makedirs(pb_new_path)

            pb_path = os.path.join(tgt_dir, algo, file + ".pb")
            future = executor.submit(execute_command, file_path, pb_path)