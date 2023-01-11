import json
import argparse
import subprocess

TESTCASE_DIR = "/home/pp22/share/.testcases/hw4"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", type=str, default="01", help="Test case number")
    return parser.parse_args()

def parse_json(filename):
    with open(filename) as f:
        return json.load(f)
    
def run_test(inputs):
    subprocess.run([f"\
        srun \
        -N{inputs['NODES']} \
        -c{inputs['CPUS']} \
        ./mapreduce \
        {inputs['JOB_NAME']} \
        {inputs['NUM_REDUCER']} \
        {inputs['DELAY']} \
        {inputs['CHUNK_SIZE']} \
        {inputs['LOCALITY_CONFIG_FILENAME']} \
        ../outputs/"
    ], shell=True)

if __name__ == "__main__":
    args = parse_args()
    inputs = parse_json(f"{TESTCASE_DIR}/{args.t}.json")
    run_test(inputs)
