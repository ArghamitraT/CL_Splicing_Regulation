#!/bin/bash

set -e
cd $HOME
source ~/.bashrc
source $HOME/miniconda3/etc/profile.d/conda.sh

conda activate hash_frag_env
cd /gpfs/commons/home/nkeung/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data/chromSplit/hashFragData