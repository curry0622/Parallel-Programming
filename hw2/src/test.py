import argparse
import subprocess

if __name__ == '__main__':
  # e.g. python3 test.py -t fast01 -o out.png -c 12 -m a
  parser = argparse.ArgumentParser()
  parser.add_argument('-t', type=str, default='fast01', help='test name, e.g. fast01')
  parser.add_argument('-o', type=str, default='out.png', help='out png, e.g. out.png')
  parser.add_argument('-c', type=str, default='12', help='core num, e.g. 12')
  parser.add_argument('-m', type=str, default='a', help='test mode, a for pthread and b for hybrid')
  args = parser.parse_args()

  f = open(file=f'../testcases/{args.t}.txt', mode='r')
  line = f.readline()
  subprocess.run(['make'], shell=True)
  subprocess.run([f'srun -n1 -c{args.c} ./hw2{args.m} {args.o} {line}'], shell=True)
  subprocess.run([f'hw2-diff ../testcases/{args.t}.png {args.o}'], shell=True)