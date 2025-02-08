import os
from os import listdir
from os.path import isfile, join
import shutil
import argparse
from consts import TranslationModels
from datetime import datetime
HOME = "/cs/usr/bareluz/gabi_labs/nematus_clean/"
LOCATIONS = ['A','B','C']

def cleanup(paths,files_to_ignore):
    # now = datetime.now()
    # dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    for path in paths:
        dst_path = path +"/backup"
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        files = [f for f in listdir(path) if isfile(join(path, f)) and f not in files_to_ignore]
        # shutil.make_archive(dst_path, 'zip',path)
        for file in files:
            # os.remove(os.path.join(path, file))
            shutil.move(os.path.join(path, file), os.path.join(dst_path, file))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--clean_embedding_table', action='store_true',
        help="weather should clean the embedding table or not")
    parser.add_argument(
        '-t', '--clean_translation_files', action='store_true',
        help="weather should clean the translation files")
    args = parser.parse_args()
    languages = ["de","ru","he","es"]
    for language in languages:
        files_to_ignore = []
        if not args.clean_embedding_table:
            files_to_ignore.append("output_translate_" + language + ".txt")

        if not args.clean_translation_files:
            for debias_method in [0,1]:
                for model in TranslationModels:
                    for location in LOCATIONS:

                        files_to_ignore+=["debiased_anti_"+str(debias_method)+"_"+model+"_"+location+".out.tmp",
                                          "non_debiased_anti_"+str(debias_method)+"_"+model+"_"+location+".out.tmp",
                                          "debiased_"+str(debias_method)+"_"+model+"_"+location+".out.tmp",
                                          "non_debiased_"+str(debias_method)+"_"+model+"_"+location+".out.tmp"]


        cleanup([HOME + "debias_outputs/en-" + language + "/debias",
                 HOME + "debias_outputs/en-" + language +"/output", ],
                files_to_ignore)
    cleanup([HOME + "mt_gender/translations/NEMATUS",
             HOME + "mt_gender/translations/EASY_NMT",
             HOME + "mt_gender/data/aggregates"],
            ["en_anti.txt", "en_pro.txt", "en.txt"])

