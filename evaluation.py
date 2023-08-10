#!/usr/bin/env python
import os
import sys
from time import time
import argparse

import numpy as np
import pandas as pd
from visor.evaluation import VISOREvaluation

default_visor = '/Volumes/SSD_WD/VISOR/VISOR_YTVOS_VAL/val_no_hand/Normal'

time_start = time()
parser = argparse.ArgumentParser()
parser.add_argument('-gt', type=str, help='Path to the VISOR folder containing the JPEGImages, Annotations, GT folders',
                    required=False, default=default_visor)
parser.add_argument('-res', type=str, help='Path to the folder containing the sequences folders',
                    required=False, default='/Volumes/SSD_WD/VISOR/VISOR_YTVOS_VAL/Results/pure_STCN')
parser.add_argument('-padding', help='Whether to pad the results or not', action='store_true')
args, _ = parser.parse_known_args()

ext = '_padding' if args.padding else '_no_padding'
csv_name_global = f'global_results{ext}.csv'
csv_name_per_sequence = f'per-sequence_results{ext}.csv'

# Check if the method has been evaluated before, if so read the results, otherwise compute the results
csv_name_global_path = os.path.join(args.res, csv_name_global)
csv_name_per_sequence_path = os.path.join(args.res, csv_name_per_sequence)
if os.path.exists(csv_name_global_path) and os.path.exists(csv_name_per_sequence_path):
    print('Using precomputed results...')
    table_g = pd.read_csv(csv_name_global_path)
    table_seq = pd.read_csv(csv_name_per_sequence_path)
else:
    # Create dataset and evaluate
    dataset_eval = VISOREvaluation(root=args.gt)

    # Evaluate the results
    print(f'Evaluating {args.res}...')
    metrics_res, uneven_seq = dataset_eval.evaluate(args.res, padding=args.padding)
    print(f'{len(uneven_seq)} sequences have different number of frames vs GT')

    J, F = metrics_res['J'], metrics_res['F']
    # Generate dataframe for the general results
    g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']
    final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
    g_res = np.array([final_mean, np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]), np.mean(F["M"]), np.mean(F["R"]),
                      np.mean(F["D"])])
    g_res = np.reshape(g_res, [1, len(g_res)])
    table_g = pd.DataFrame(data=g_res, columns=g_measures)
    with open(csv_name_global_path, 'w') as f:
        table_g.to_csv(f, index=False, float_format="%.3f")
    print(f'Global results saved in {csv_name_global_path}')

    # Generate a dataframe for the per sequence results
    seq_names = list(J['M_per_object'].keys())
    seq_measures = ['Sequence', 'J-Mean', 'F-Mean']
    J_per_object = [J['M_per_object'][x] for x in seq_names]
    F_per_object = [F['M_per_object'][x] for x in seq_names]
    table_seq = pd.DataFrame(data=list(zip(seq_names, J_per_object, F_per_object)), columns=seq_measures)
    table_seq = table_seq.sort_values(by=['Sequence'], ascending=True)
    with open(csv_name_per_sequence_path, 'w') as f:
        table_seq.to_csv(f, index=False, float_format="%.3f")
    print(f'Per-sequence results saved in {csv_name_per_sequence_path}')

# Print the results
print(f"----------------- Global results for VISOR val ----------------")
print(table_g.to_string(index=False))
total_time = time() - time_start
print('\nTotal time: ' + str(total_time))
