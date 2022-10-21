import numpy as np
import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', default='01', help='input file num')
  args = parser.parse_args()

  FILE_IN = f'/home/pp22/share/hw1/testcases/{args.input}.out'
  FILE_OUT = '../sample/out'

  fin = np.fromfile(FILE_IN,  dtype=np.float32)
  fout = np.fromfile(FILE_OUT,  dtype=np.float32)

  print(f'fin: {fin.size}, fout: {fout.size}')

  if(fin.size != fout.size):
    print('[ERROR] file size different')

  for i in range(fin.size):
    if fin[i] != fout[i]:
      print(f'[ERROR] on data[{i}], expect: {fin[i]}, got: {fout[i]}')
      break
    elif i == fin.size - 1:
      print('[Accepted]')