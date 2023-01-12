import json
import argparse
import subprocess

TESTCASE_DIR = "/home/pp22/share/.testcases/hw4"
OUTPUT_DIR = "../outputs"
EXEC = "./mapreduce"

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
        {EXEC} \
        {inputs['JOB_NAME']} \
        {inputs['NUM_REDUCER']} \
        {inputs['DELAY']} \
        {TESTCASE_DIR}/{inputs['INPUT_FILE_NAME']} \
        {inputs['CHUNK_SIZE']} \
        {TESTCASE_DIR}/{inputs['LOCALITY_CONFIG_FILENAME']} \
        {OUTPUT_DIR}"
    ], shell=True)

if __name__ == "__main__":
    args = parse_args()
    inputs = parse_json(f"{TESTCASE_DIR}/{args.t}.json")
    run_test(inputs)
