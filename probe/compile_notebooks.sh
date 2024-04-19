#!/usr/bin/env bash
# main.sh

# files=(2024-02-03*)
# files=( 2024-02-03_A-particles.ipynb  2024-02-03_B-particles_L1.ipynb  2024-02-03_C-particles_coverage.ipynb  2024-02-03_D-trames-particles.ipynb  2024-02-03_E-trames-particles-interférences.ipynb)

# for file in "${files[@]}"; do
for file in "$@"; do
    echo "Running ${file}"
    jupyter nbconvert  --ExecutePreprocessor.timeout=0 --allow-errors --execute --to notebook --inplace ${file}
done

