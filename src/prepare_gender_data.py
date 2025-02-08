import pickle

import sys
# sys.path.append("..") # Adds higher directory to python modules path.
sys.path.append("../../debias_files") # Adds higher directory to python modules path.

from consts import get_evaluate_gender_files, LANGUAGE_STR_MAP, parse_config, Language
import argparse
from detokenize import detokenize_matrix
def prepare_gender_sents_translation_to_evaluation(source_filename,source_translated_filename, dest_filename, language):
    """
    given source filename with sentences in the source language,
    and source_translated_filename with the translations to the dest language
    creates a new file dest_filename with the following format
    <sentence> ||| <translation> \n
    :param source_filename: source filename with sentences in the source language
    :param source_translated_filename: file with the translations to the dest language
    :param dest_filename: the file with the resulted format
    """
    print("source_filename")
    print(source_filename)
    print("source_translated_filename")
    print(source_translated_filename)
    with open(source_filename, "r") as s1,open(source_translated_filename, "r") as s2, open(dest_filename, "w") as d:
        lines_source = s1.readlines()
        # lines_translated = detokenize_matrix(s2.readlines(), LANGUAGE_STR_MAP[Language(language)])
        lines_translated = s2.readlines()
        for line_source, line_translated in zip(lines_source,lines_translated):
            d.write(line_source.strip().split("\t")[2] + " ||| " + line_translated.rstrip() + "\n")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '-c', '--config_str', type=str, required=True,
            help="a config dictionary str that conatains: \n"
             "USE_DEBIASED= run translate on the debiased dictionary or not\n"
             "LANGUAGE= the language to translate to from english. RUSSIAN = 0, GERMAN = 1, HEBREW = 2\n"
             "DEBIAS_METHOD= the debias method. HARD_DEBIAS = 0 INPL = 1\n"
             "TRANSLATION_MODEL= the translation model. Nematus=0, EasyNMT=1\n"
             "DEBIAS_ENCODER= whether to debias the encoder\n"
             "BEGINNING_DECODER_DEBIAS= whether to debias the inputs of the decoder\n"
             "END_DECODER_DEBIAS= whether to debias the outputs of the decoder\n"""
             "WORDS_TO_DEBIAS= set of words to debias. ALL_VOCAB = 0, ONE_TOKEN_PROFESSIONS = 1, ALL_PROFESSIONS = 2 \n")
    args = parser.parse_args()
    ANTI_TRANSLATED_DEBIASED, ANTI_TRANSLATED_NON_DEBIASED, DEBIASED_EVAL, \
    NON_DEBIASED_EVAL, EN_ANTI_MT_GENDER, EN_NEUTRAL_MT_GENDER = get_evaluate_gender_files(args.config_str)
    # detokenize_file(EN_ANTI_MERGED,EN_ANTI_MERGED_DETOKENIZED)
    # parse_gender_sents(EN_ANTI, EN_ANTI_PARSED)
    prepare_gender_sents_translation_to_evaluation(EN_ANTI_MT_GENDER, ANTI_TRANSLATED_DEBIASED, DEBIASED_EVAL, parse_config(args.config_str)["LANGUAGE"])
    prepare_gender_sents_translation_to_evaluation(EN_ANTI_MT_GENDER, ANTI_TRANSLATED_NON_DEBIASED, NON_DEBIASED_EVAL, parse_config(args.config_str)["LANGUAGE"])