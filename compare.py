# ==============================================================================
# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Erica Cooper
# All rights reserved.
# ==============================================================================

import os
import argparse
import numpy as np
import scipy.stats
from collections import defaultdict


def extract_systemID(uttID):
    return uttID.split("-")[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--answer_file_a",
        type=str,
        required=True,
        help="Path to answer.txt to compare.",
    )
    parser.add_argument(
        "--answer_file_b",
        type=str,
        required=True,
        help="Path to answer.txt to compare.",
    )
    parser.add_argument(
        "--datadir", type=str, required=True, help="Path of your DATA/ directory"
    )
    args = parser.parse_args()

    system_mos_path = os.path.join(args.datadir, "mydata_system.csv")
    utterance_mos_path = os.path.join(args.datadir, "sets/val_mos_list.txt")

    answer_a_by_uttID = {}
    answer_a_by_systemID = defaultdict(list)
    with open(args.answer_file_a, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            uttID = parts[0]
            systemID = extract_systemID(uttID)
            MOS = float(parts[1])
            answer_a_by_uttID[uttID] = MOS
            answer_a_by_systemID[systemID].append(MOS)

    avg_answer_a_by_systemID = {}
    for systemID, MOSs in answer_a_by_systemID.items():
        avg_answer_a_by_systemID[systemID] = np.mean(MOSs)

    answer_b_by_uttID = {}
    answer_b_by_systemID = defaultdict(list)
    with open(args.answer_file_b, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            uttID = parts[0]
            systemID = extract_systemID(uttID)
            MOS = float(parts[1])
            answer_b_by_uttID[uttID] = MOS
            answer_b_by_systemID[systemID].append(MOS)

    avg_answer_b_by_systemID = {}
    for systemID, MOSs in answer_b_by_systemID.items():
        avg_answer_b_by_systemID[systemID] = np.mean(MOSs)

    # ## compute correls

    # ### UTTERANCE

    utterance_mos_a = []
    utterance_mos_b = []

    with open(utterance_mos_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            uttID = parts[0].split(".")[0]
            utterance_mos_a.append(answer_a_by_uttID[uttID])
            utterance_mos_b.append(answer_b_by_uttID[uttID])

    utterance_mos_a = np.array(utterance_mos_a)
    utterance_mos_b = np.array(utterance_mos_b)

    MSE = np.mean((utterance_mos_a - utterance_mos_b) ** 2)
    print("[UTTERANCE] Test error= %f" % MSE)
    LCC = np.corrcoef(utterance_mos_a, utterance_mos_b)
    print("[UTTERANCE] Linear correlation coefficient= %f" % LCC[0][1])
    SRCC = scipy.stats.spearmanr(utterance_mos_a.T, utterance_mos_b.T)
    print("[UTTERANCE] Spearman rank correlation coefficient= %f" % SRCC[0])
    KTAU = scipy.stats.kendalltau(utterance_mos_a, utterance_mos_b)
    print("[UTTERANCE] Kendall Tau rank correlation coefficient= %f" % KTAU[0])

    # ### SYSTEM

    system_mos_a = []
    system_mos_b = []

    with open(system_mos_path, "r") as f:
        f.readline()  # ignore first line
        for line in f:
            parts = line.strip().split(",")
            systemID = parts[0]
            system_mos_a.append(avg_answer_a_by_systemID[systemID])
            system_mos_b.append(avg_answer_b_by_systemID[systemID])

    system_mos_a = np.array(system_mos_a)
    system_mos_b = np.array(system_mos_b)

    MSE = np.mean((system_mos_a - system_mos_b) ** 2)
    print("[SYSTEM] Test error= %f" % MSE)
    LCC = np.corrcoef(system_mos_a, system_mos_b)
    print("[SYSTEM] Linear correlation coefficient= %f" % LCC[0][1])
    SRCC = scipy.stats.spearmanr(system_mos_a.T, system_mos_b.T)
    print("[SYSTEM] Spearman rank correlation coefficient= %f" % SRCC[0])
    KTAU = scipy.stats.kendalltau(system_mos_a, system_mos_b)
    print("[SYSTEM] Kendall Tau rank correlation coefficient= %f" % KTAU[0])


if __name__ == "__main__":
    main()
