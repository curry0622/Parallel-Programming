import os
import subprocess

if __name__ == '__main__':
    DIR = '/tmp/dataset-nthu-pp22/pp22/share/lab3/testcases/'
    files = os.listdir(DIR)
    for f in files:
        print(f)
        subprocess.run([f'ln -s {DIR}{f}'], shell=True, cwd='./testcases/')