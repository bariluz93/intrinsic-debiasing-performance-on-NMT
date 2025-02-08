import argparse
from easynmt import EasyNMT
import torch
import os
from sacrebleu.metrics import BLEU
bleu = BLEU()
os.environ['TRANSFORMERS_CACHE'] = '/cs/snapless/gabis/bareluz'
from consts import get_basic_configurations, LANGUAGE_STR_MAP, Language,LANGUAGE_CODES_MAP
device = "cuda" if torch.cuda.is_available() else "cpu"

def translate(input_file:str, output_file:str, config:str):
    USE_DEBIASED, LANGUAGE, _, DEBIAS_METHOD, TRANSLATION_MODEL ,DEBIAS_ENCODER,BEGINNING_DECODER_DEBIAS,END_DECODER_DEBIAS,WORDS_TO_DEBIAS= get_basic_configurations(config)
    with open(input_file, 'r') as input, open(output_file, 'w') as output:
        model = EasyNMT('mbart50_en2m', src_lang = "en_XX",tgt_lang = LANGUAGE_CODES_MAP[LANGUAGE_STR_MAP[Language(LANGUAGE)]])
        translations = model.translate(input.readlines(), source_lang='en_XX',
                                       target_lang=LANGUAGE_STR_MAP[Language(LANGUAGE)],
                                       show_progress_bar=True,
                                       use_debiased = USE_DEBIASED,
                                       debias_method=DEBIAS_METHOD,
                                       debias_encoder=DEBIAS_ENCODER,
                                       beginning_decoder_debias=BEGINNING_DECODER_DEBIAS,
                                       end_decoder_debias=END_DECODER_DEBIAS,
                                       words_to_debias=WORDS_TO_DEBIAS,
                                       translation_model = TRANSLATION_MODEL)
        output.writelines(translations)

# def print_bleu(output_file,refs_path):
#     with open(refs_path,'r') as f, open(output_file,'r') as f2:
#         refs = f.readlines()
#         translations =f2.readlines()
#     print("bleu: ")
#     print(bleu.corpus_score(translations, [refs]))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', type=str,
        help="input file")
    parser.add_argument(
        '-o', '--output', type=str,
        help="output file")
    parser.add_argument(
        '-c', '--config_str', type=str, required=True,
        help="a config dictionary str that conatains: \n"
             "USE_DEBIASED= run translate on the debiased dictionary or not\n"
             "LANGUAGE= the language to translate to from english. RUSSIAN = 0, GERMAN = 1, HEBREW = 2\n"
             "DEBIAS_METHOD= the debias method. HARD_DEBIAS = 0 INLP = 1\n"
             "TRANSLATION_MODEL= the translation model. Nematus=0, EasyNMT=1\n"
             "DEBIAS_ENCODER= whether to debias the encoder\n"
             "BEGINNING_DECODER_DEBIAS= whether to debias the inputs of the decoder\n"
             "END_DECODER_DEBIAS= whether to debias the outputs of the decoder\n"""
             "WORDS_TO_DEBIAS= set of words to debias. ALL_VOCAB = 0, ONE_TOKEN_PROFESSIONS = 1, ALL_PROFESSIONS = 2 \n"

    )
    args = parser.parse_args()
    translate(args.input, args.output, args.config_str)
    # print_bleu(args.output,refs_path_ru)
