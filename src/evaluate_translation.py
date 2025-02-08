from sacrebleu.metrics import BLEU
from consts import get_evaluate_translation_files, LANGUAGE_STR_MAP, parse_config, Language, get_basic_configurations, TranslationModelsEnum
from detokenize import detokenize_matrix
import argparse
def evaluate_translation(language):
    # refs = []
    with open(TRANSLATED_DEBIASED, 'r') as d, open(TRANSLATED_NON_DEBIASED, 'r') as nd, open(
            BLEU_GOLD_DATA, 'r') as gt, open(BLEU_GOLD_DATA_NON_TOKENIZED,'r') as gnt:
        sys_debiased = d.readlines()
        sys_non_debiased = nd.readlines()
        refs = gnt.readlines()

        if int(TRANSLATION_MODEL) == TranslationModelsEnum.NEMATUS.value:
            sys_debiased = detokenize_matrix(sys_debiased, LANGUAGE_STR_MAP[Language(int(language))])
            sys_non_debiased = detokenize_matrix(sys_non_debiased, LANGUAGE_STR_MAP[Language(int(language))])
            refs = gt.readlines()
            refs = detokenize_matrix(refs, LANGUAGE_STR_MAP[Language(int(language))])
    bleu = BLEU()
    print("debiased")
    print(bleu.corpus_score(sys_debiased, [refs]))
    print("non debiased")
    print(bleu.corpus_score(sys_non_debiased, [refs]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '-c', '--config_str', type=str, required=True,
            help="a config dictionary str that conatains: \n"
                 "debiased= run translate on the debiased dictionary or not\n"
                 "language= the language to translate to from english. RUSSIAN = 0, GERMAN = 1, HEBREW = 2\n"
                 "collect_embedding_table= run translate to collect embedding table or not\n"
                 "print_line_nums= whether to print line numbers to output file in translate")
    args = parser.parse_args()
    BLEU_GOLD_DATA,BLEU_GOLD_DATA_NON_TOKENIZED,_, TRANSLATED_DEBIASED, TRANSLATED_NON_DEBIASED = get_evaluate_translation_files(args.config_str)
    _, _, _, _, TRANSLATION_MODEL,_,_,_,_ = get_basic_configurations(args.config_str)
    evaluate_translation( parse_config(args.config_str)["LANGUAGE"])