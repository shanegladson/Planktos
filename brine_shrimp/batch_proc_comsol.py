'''
Batch script for running multiple simulations
'''

from pathlib import Path
import os

def main():
    p = Path('./data/comsol')
    data_files = [x for x in p.iterdir() if x.is_file() and x.suffix == '.vtu']
    for f in data_files:
        prefix = 'Velocity_'
        suffix = '.vtu'
        name = f.name[len(prefix):]
        name = name[:-len(suffix)] + '_var5_N100000'
        cmd = 'python comsol_shrimp.py -N 100000 -o {} -d {}'.format(name, f)
        print('Now processing {}.'.format(f))
        os.system(cmd)