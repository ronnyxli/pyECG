
'''
Usage:
    deploy_at_sims_batch.py <data_dir>
                            [rec_list=<rl>]
    deploy_at_sims_batch.py -h | --help

Options:
    -h --help               Show this documentation
'''

import docopt
import os
import glob

import pdb



def main(args):

    header_files = [x for x in os.listdir(data_dir) if os.path.splitext(x)[1] == '.hea']

    # loop all header files
    for hf in header_files:
        pdb.set_trace()

if __name__ == "__main__":
    main(docopt(__doc__))
