import argparse
import re
from consts import LANGUAGE_STR_MAP, DebiasMethod, DEBIAS_FILES_HOME, OUTPUTS_HOME, TranslationModels
import pandas as pd
import numpy as np
import json
import csv
import math
from datetime import datetime
from pathlib import Path
import ast
import glob
import subprocess

ANTI_ACCURACY_IDX = 0
F1_MALE_IDX = 1
F1_FEMALE_IDX = 2
DELTA_G_IDX = 3

ANTI_ACCURACY = "Anti Accuracy"
GENDER_RESULTS = "Gender Results"
DELTA_G ="Delta G"
BLEU = "Bleu"

INLP = "INLP"

HARD_DEBIAS = "Hard Debias"

LANGUAGES = LANGUAGE_STR_MAP.values()
LANGUAGES = list(LANGUAGES)
LANGUAGES.remove('es')
# LANGUAGES.remove('he')
# LANGUAGES.remove('de')
LANGUAGES.remove('ru')
DEBIAS_METHODS = [d.value for d in DebiasMethod]
DEBIAS_LOCS = ["A", "B", "C","A_B","A_C","B_C","A_B_C"]

DEBIAS_WORDS_SETS={0:"all_dict",1:'1_tokens_professions',2:"all professions"}

def get_translation_results(file_name):
    result = {}
    if not Path(file_name).is_file():
        result["debiased"] = None
        result["non_debiased"] = None
    else:
        with open(file_name) as f:
            lines = f.readlines()
            debiased_line = lines.index("debiased\n") + 1
            result["debiased"] = float(lines[debiased_line].split(" ")[2])
            non_debiased_line = lines.index("non debiased\n") + 1
            result["non_debiased"] = float(lines[non_debiased_line].split(" ")[2])
    return result


def get_gender_results(file_name):
    result = {}
    if not Path(file_name).is_file():
        result["debiased"] = None
        result["non_debiased"] = None
        professions_accuracies_debiased=None
        professions_accuracies_non_debiased=None
    else:
        with open(file_name) as f:
            line = f.readline()
            line_num = 1
            debiased_found = False
            non_debiased_found = False

            while line:
                if line.__contains__("*debiased results*"):
                    debiased_found = True
                if line.__contains__("*non debiased results*"):
                    non_debiased_found = True
                # get general accuracy
                match = re.search(
                    "{\"acc\": (.*), \"f1_male\": (.*), \"f1_female\": (.*), \"unk_male\": .*, \"unk_female\": .*, \"unk_neutral\": .*}",
                    line)
                if match:
                    accuracy = float(match.group(1))
                    f1_male,f1_female = float(match.group(2)),float(match.group(3))
                    delta_g = f1_male-f1_female
                    if debiased_found:
                        result["debiased"] = [accuracy,f1_male,f1_female,delta_g]
                    elif non_debiased_found:
                        result["non_debiased"] = [accuracy,f1_male,f1_female,delta_g]
                # get professions accuracy
                if line.__contains__("prof_accuracies"):
                    line = f.readline()
                    line_num += 1
                    if debiased_found:
                        professions_accuracies_debiased = ast.literal_eval(line)
                        debiased_found = False
                    elif non_debiased_found:
                        professions_accuracies_non_debiased = ast.literal_eval(line)
                        non_debiased_found = False
                line = f.readline()
                line_num += 1
    return result, professions_accuracies_debiased, professions_accuracies_non_debiased


def get_all_results(result_files):
    results = {}
    professions_results = {}
    for language in LANGUAGES:
        results[language] = {BLEU: {}, GENDER_RESULTS: {}}
        for debias_method in DEBIAS_METHODS:
            for loc in DEBIAS_LOCS:
                results[language][BLEU][(debias_method, loc)] = \
                    get_translation_results(result_files[language][BLEU][(debias_method, loc)])

                results[language][GENDER_RESULTS][(debias_method, loc)],\
                professions_accuracies_debiased,\
                professions_accuracies_non_debiased = \
                get_gender_results(result_files[language][GENDER_RESULTS][(debias_method, loc)])

                if professions_accuracies_debiased is not None and professions_accuracies_non_debiased is not None:
                    for p in professions_accuracies_debiased.keys():
                        if p not in professions_results:
                            professions_results[p] = {}
                        if language not in professions_results[p]:
                            professions_results[p][language] = {}

                        professions_results[p][language][debias_method] = professions_accuracies_debiased[p] - \
                                                                      professions_accuracies_non_debiased[p]
    # print(json.dumps(results, sort_keys=True, indent=4))
    results_table={}
    for language in LANGUAGES:
        results_table[language] = {BLEU: [], ANTI_ACCURACY: [], DELTA_G: []}

        for loc in DEBIAS_LOCS:
            ### get bleu results
            loc_bleu = []
            orig_bleu = results[language][BLEU][(DEBIAS_METHODS[0], loc)]["non_debiased"]
            loc_bleu.append(orig_bleu)
            for debias_method in DEBIAS_METHODS:
                method_bleu = results[language][BLEU][(debias_method, loc)]["debiased"]
                loc_bleu.append(method_bleu)
            if None in loc_bleu:
                results_table[language][BLEU].append(loc_bleu)
            else:
                results_table[language][BLEU].append(np.around(loc_bleu,2))

            ### get anti accuracy results
            loc_anti_accuracy = []
            if results[language][GENDER_RESULTS][(DEBIAS_METHODS[0], loc)]["non_debiased"] is None:
                orig_anti_accuracy =None
            else:
                orig_anti_accuracy = results[language][GENDER_RESULTS][(DEBIAS_METHODS[0], loc)]["non_debiased"][ANTI_ACCURACY_IDX]
            loc_anti_accuracy.append(orig_anti_accuracy)
            for debias_method in DEBIAS_METHODS:
                if results[language][GENDER_RESULTS][(debias_method, loc)]["non_debiased"] is None:
                    method_anti_accuracy = None
                else:
                    method_anti_accuracy = results[language][GENDER_RESULTS][(debias_method, loc)]["debiased"][ANTI_ACCURACY_IDX]
                loc_anti_accuracy.append(method_anti_accuracy)
            if None in loc_anti_accuracy:
                results_table[language][ANTI_ACCURACY].append(loc_anti_accuracy)
            else:
                results_table[language][ANTI_ACCURACY].append(np.around(loc_anti_accuracy,2))

            ### get delta g results
            loc_delta_g = []
            if results[language][GENDER_RESULTS][(DEBIAS_METHODS[0], loc)]["non_debiased"] is None:
                loc_delta_g = [None,None,None,None,None,None,None,None,None,]
            else:
                orig_f1_male = results[language][GENDER_RESULTS][(DEBIAS_METHODS[0], loc)]["non_debiased"][F1_MALE_IDX]
                orig_f1_female = results[language][GENDER_RESULTS][(DEBIAS_METHODS[0], loc)]["non_debiased"][F1_FEMALE_IDX]
                orig_delta_g = results[language][GENDER_RESULTS][(DEBIAS_METHODS[0], loc)]["non_debiased"][DELTA_G_IDX]
                loc_delta_g.append(orig_f1_male)
                loc_delta_g.append(orig_f1_female)
                loc_delta_g.append(orig_delta_g)

                for debias_method in DEBIAS_METHODS:
                    method_f1_male = results[language][GENDER_RESULTS][(debias_method, loc)]["debiased"][F1_MALE_IDX]
                    method_f1_female = results[language][GENDER_RESULTS][(debias_method, loc)]["debiased"][F1_FEMALE_IDX]
                    method_delta_g = results[language][GENDER_RESULTS][(debias_method, loc)]["debiased"][DELTA_G_IDX]
                    loc_delta_g.append(method_f1_male)
                    loc_delta_g.append(method_f1_female)
                    loc_delta_g.append(method_delta_g)
            if None in loc_delta_g:
                results_table[language][DELTA_G].append(loc_delta_g)
            else:
                results_table[language][DELTA_G].append(np.around(loc_delta_g,2))


    professions_results_table = {}
    for p in professions_results.keys():
        professions_results_table[p] = [professions_results[p][language][debias_method] for language in LANGUAGES for
                                        debias_method in DEBIAS_METHODS]

    return results_table, professions_results_table


def write_professions_results_to_csv(professions_results, model):
    headers = [None, "Russian", None, "German", None, "Hebrew", None]
    sub_headers = [None] + [HARD_DEBIAS, INLP] * 3
    index = [[p] for p in professions_results.keys()]
    data = np.append(index, list(professions_results.values()), axis=1)
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    Path(DEBIAS_FILES_HOME + "results").mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS_HOME + "results/professions_results_" + model + "_" + dt_string + ".csv", 'w', encoding='UTF8',
              newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerow(sub_headers)
        writer.writerows(data)

def write_results_to_dir(results, model,words_to_debias):
    # created dir with the current time in the results directory and write each results table for each language and
    # for bleu and anti accuracy to different file in this folder
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    # Path(OUTPUTS_HOME + "results").mkdir(parents=True, exist_ok=True)
    dir_path =OUTPUTS_HOME + "results/"+dt_string+"_"+words_to_debias
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    for language in LANGUAGES:
           path_bleu =  dir_path+"/results_" + model + "_" + language+"_"+"bleu" + ".csv"
           write_results_to_csv(results[language][BLEU],path_bleu)
           path_anti_accuracy =  dir_path+"/results_" + model + "_" + language+"_"+"anti_accuracy" + ".csv"
           write_results_to_csv(results[language][ANTI_ACCURACY],path_anti_accuracy)
           path_delta_g =  dir_path+"/results_" + model + "_" + language+"_"+"delta_g" + ".csv"
           write_results_to_csv_f1_delta_g(results[language][DELTA_G],path_delta_g)

def write_results_to_csv_f1_delta_g(results,path):
    # writes the current language results to a csv file given by path
    headers = [None,"Orig",None,None,HARD_DEBIAS,None,None,INLP,None,None]
    sub_headers = [None] + ["F1 Male", "F1 Female", "Delta G"] * 3
    index = [["Encoder Input"],["Decoder Input"],["Decoder Output"],["Encoder Input+Decoder Input"],
             ["Encoder Input+Decoder Output"],["Decoder Input+Decoder Output"],["Encoder Input+Decoder Input+Decoder Output"]]
    data = np.append(index, results, axis=1)
    with open(path, 'w', encoding='UTF8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerow(sub_headers)
        writer.writerows(data)
def write_results_to_csv(results,path):
    # writes the current language results to a csv file given by path
    headers = [None,"Orig",HARD_DEBIAS,INLP]
    # sub_headers = [None] + ["Bleu", "delta Bleu", "anti accuracy", "delta anti accuracy"] * 3
    index = [["Encoder Input"],["Decoder Input"],["Decoder Output"],["Encoder Input+Decoder Input"],
             ["Encoder Input+Decoder Output"],["Decoder Input+Decoder Output"],["Encoder Input+Decoder Input+Decoder Output"]]
    # index = [["Original"], [HARD_DEBIAS], [INLP]]
    data = np.append(index, results, axis=1)
    # now = datetime.now()
    # dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    # Path(DEBIAS_FILES_HOME + "results").mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='UTF8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        # writer.writerow(sub_headers)
        writer.writerows(data)


def write_results_to_latex(results_dir):
    files=glob.glob(results_dir+"*.csv")
    for f in files:
        file_name=Path(f).stem



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help="the translation model:\n"
             "0 = Nematus\n1 = Easy NMT")
    parser.add_argument(
        '-w', '--words_to_debias', type=str, required=True,
        help="the set of words to debias:\n"
             "0 = all dictionary\n1 = 1 token professions\n2 = all professions")
    args = parser.parse_args()
    model = TranslationModels[int(args.model)]
    LANGUAGES = list(LANGUAGES)
    # LANGUAGES.remove('es')
    result_files = {}

    for language in LANGUAGES:
        result_files[language] = {BLEU: {}, GENDER_RESULTS: {}}
        for debias_method in DEBIAS_METHODS:
            for loc in DEBIAS_LOCS:
                result_files[language][BLEU][(debias_method, loc)] = \
                    OUTPUTS_HOME + "en-" + language + "/debias/translation_evaluation_" + language + "_" + str(
                        debias_method) + "_" + model + "_" + loc + ".txt"
                result_files[language][GENDER_RESULTS][(debias_method, loc)] = \
                    OUTPUTS_HOME + "en-" + language + "/debias/gender_evaluation_" + language + "_" + str(
                        debias_method) + "_" + model + "_" + loc + ".txt"

    # for language in LANGUAGES:
    #     for debias_method in DEBIAS_METHODS:
    #         result_files[language][debias_method] = {}
    #         result_files[language][debias_method]["translation"] = OUTPUTS_HOME + "en-" + language + "/debias/translation_evaluation_"+ language + "_"+str(debias_method)+"_"+model+".txt"
    #         result_files[language][debias_method]["gender"] = OUTPUTS_HOME + "en-" + language + "/debias/gender_evaluation_" + language + "_"+str(debias_method)+"_"+model+".txt"
    res, professions_results_table = get_all_results(result_files)
    write_results_to_dir(res, model,DEBIAS_WORDS_SETS[int(args.words_to_debias)] )
    write_professions_results_to_csv(professions_results_table, model)
