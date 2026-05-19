#!/bin/bash
# JOB HEADERS HERE
#SBATCH --job-name=test_job
#SBATCH --output=test_job.%j.out
#SBATCH --error=test_job.%j.err
#SBATCH --time=47:00:00
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=50GB

set -e
ml python/3.9.0
PROJECT_DIR="$SCRATCH/rcl_market_assessment"

rm -rf "$PROJECT_DIR"
git clone https://github.com/afsuarezg/rcl-market-assessment.git "$PROJECT_DIR"

# pip install any dependencies if needed:
# pip install --user pyblp pandas

python3 <<EOF
import os
import sys

project_dir = os.path.join(os.environ['SCRATCH'], 'rcl_market_assessment')
sys.path.insert(0, project_dir)
os.chdir(project_dir)

import blp_blp
import main
import nevo_blp

#Se han quitado ['sugar'] y ['mushy'] porque ya corrieron pero tuvieron problemas de colinealidad frente a ['income', 'income_squared', 'age', 'child']
nevo_blp.main(
    x2_combos=[['sugar', 'mushy'], ['sugar'], ['mushy']], 
    demo_combos=[['income', 'age'] ,['income', 'child']   ,['income', 'income_squared'],  ['income', 'income_squared', 'age'], 
    ['income', 'age', 'child'], ['income', 'income_squared', 'age'], ['income', 'income_squared', 'child'] ,['income', 'income_squared', 'age', 'child']],
            target_seeds=40)

blp_blp.main(x2_combos=[['hpwt', 'air'], ['hpwt', 'space'],['mpd', 'space'], ['hpwt', 'air', 'mpd'], ['hpwt', 'air', 'space']  ,['hpwt', 'mpd', 'space'], ['hpwt', 'air', 'mpd', 'space']],
    demo_combos=[['I(1 / income)']],
    target_seeds=30)



EOF
