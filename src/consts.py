import json
import ast

PROJECT_HOME = "/cs/labs/gabis/bareluz/nematus_clean/"
DEBIAS_FILES_HOME = "/cs/usr/bareluz/gabi_labs/nematus_clean/debias_files/"
MT_GENDER_HOME = "/cs/usr/bareluz/gabi_labs/nematus_clean/mt_gender/"
DATA_HOME="/cs/snapless/gabis/bareluz/debias_nmt_data/"
# DATA_HOME = "/cs/snapless/gabis/bareluz/debias_nmt_data/data/"
# ANTI_DATA_HOME = "/cs/snapless/gabis/bareluz/debias_nmt_data/anti_data/"
# ANNOTATIONS_DATA_HOME = "/cs/snapless/gabis/bareluz/debias_nmt_data/professions_annotations/"
OUTPUTS_HOME = "/cs/usr/bareluz/gabi_labs/nematus_clean/debias_outputs/"
ENGLISH_VOCAB = '/cs/snapless/gabis/bareluz/debias_nmt_data/en_vocab_merged.txt'

from enum import Enum
from datetime import datetime
lang_to_gender_specific_words_map = {'ru':['он','она'],'de':['er','sie'],'he':['הוא','היא'],'en':['he','she'],'es':['él','ella']}
class Language(Enum):
    RUSSIAN = 0
    GERMAN = 1
    HEBREW = 2
    SPANISH = 3

TranslationModels =["NEMATUS","EASY_NMT"]
class TranslationModelsEnum(Enum):
    NEMATUS = 0
    EASY_NMT = 1
LANGUAGE_STR_TO_INT_MAP = {'ru': 0,'de':1,'he':2, 'es':3}
LANGUAGE_STR_MAP = {Language.RUSSIAN: "ru", Language.GERMAN: "de", Language.HEBREW: "he", Language.SPANISH: "es"}
LANGUAGE_OPPOSITE_STR_MAP = {"ru": Language.RUSSIAN , "de": Language.GERMAN, "he":Language.HEBREW, "es":Language.SPANISH}

locations_map={"DEBIAS_ENCODER":"_A","BEGINNING_DECODER_DEBIAS":"_B","END_DECODER_DEBIAS":"_C"}
class DebiasMethod(Enum):
    HARD_DEBIAS = 0
    INLP = 1
class WordsToDebias(Enum):
    ALL_VOCAB = 0
    ONE_TOKEN_PROFESSIONS = 1
    ALL_PROFESSIONS = 2

EMBEDDING_SIZE = 256

param_dict = {
    Language.RUSSIAN:
        {
            "DICT_SIZE": 30648,
            "ENG_DICT_FILE": DATA_HOME+"data/" + "en_ru_30.11.20//train.clean.unesc.tok.tc.bpe.en.json",
            "BLEU_GOLD_DATA": DATA_HOME+"data/" + "en_ru_30.11.20/newstest2019-enru.unesc.tok.tc.bpe.ru",
            "BLEU_GOLD_DATA_NON_TOKENIZED": DATA_HOME+"data/" + "en_ru_30.11.20/newstest2019-enru.ru",
            "BLEU_GOLD_DATA_NON_TOKENIZED_EN": DATA_HOME+"data/" + "en_ru_30.11.20/newstest2019-enru.en",
            "VOCAB": DATA_HOME+"data/" + "en_ru_30.11.20/ru_vocab.txt",
            "VOCAB_INLP": DATA_HOME+"data/" + "en_ru_30.11.20/ru_vocab_inlp.txt",
            "VOCAB_INLP_EN": DATA_HOME+"data/" + "en_ru_30.11.20/en_vocab_inlp.txt",
            "VOCAB_EN": DATA_HOME+"data/" + "en_ru_30.11.20/en_vocab.txt",
        },
    Language.GERMAN:
        {
            "DICT_SIZE": 29344,
            "ENG_DICT_FILE": DATA_HOME +"data/" + "en_de_5.8/train.clean.unesc.tok.tc.bpe.en.json",
            "BLEU_GOLD_DATA": DATA_HOME +"data/" + "en_de_5.8/newstest2012.unesc.tok.tc.bpe.de",
            "BLEU_GOLD_DATA_NON_TOKENIZED": DATA_HOME +"data/" + "en_de_5.8/newstest2012.de",
            "BLEU_GOLD_DATA_NON_TOKENIZED_EN": DATA_HOME +"data/" + "en_de_5.8/newstest2012.en",
            "VOCAB": DATA_HOME +"data/" + "en_de_5.8/de_vocab.txt",
            "VOCAB_INLP": DATA_HOME +"data/" + "en_de_5.8/de_vocab_inlp.txt",
            "VOCAB_INLP_EN": DATA_HOME +"data/" + "en_de_5.8/en_vocab_inlp.txt",
            "VOCAB_EN": DATA_HOME +"data/" + "en_de_5.8/en_vocab.txt",

        },
    Language.HEBREW:
        {
            "DICT_SIZE": 30545,
            "ENG_DICT_FILE": DATA_HOME +"data/" + "en_he_20.07.21//train.clean.unesc.tok.tc.bpe.en.json",
            "BLEU_GOLD_DATA": DATA_HOME +"data/" + "en_he_20.07.21//dev.unesc.tok.bpe.he",
            "BLEU_GOLD_DATA_NON_TOKENIZED": DATA_HOME +"data/" + "en_he_20.07.21//dev.he",
            "BLEU_GOLD_DATA_NON_TOKENIZED_EN": DATA_HOME +"data/" + "en_he_20.07.21//dev.en",
            "VOCAB": DATA_HOME +"data/" + "en_he_20.07.21/he_vocab.txt",
            "VOCAB_INLP": DATA_HOME +"data/" + "en_he_20.07.21/he_vocab_inlp.txt",
            "VOCAB_INLP_EN": DATA_HOME +"data/" + "en_he_20.07.21/en_vocab_inlp.txt",
            "VOCAB_EN": DATA_HOME +"data/" + "en_he_20.07.21/en_vocab.txt",

        },
    Language.SPANISH:
        {
            "DICT_SIZE": None,
            "ENG_DICT_FILE": None,
            "BLEU_GOLD_DATA": DATA_HOME +"data/" + "en_es//books.es",
            "BLEU_GOLD_DATA_NON_TOKENIZED": DATA_HOME +"data/" + "en_es//books.es",
            "BLEU_GOLD_DATA_NON_TOKENIZED_EN": DATA_HOME +"data/" + "en_es//books.en",
            "VOCAB":  DATA_HOME +"data/" + "en_es/es_vocab.txt",
            "VOCAB_INLP":DATA_HOME +"data/" + "en_es/es_vocab_inlp.txt",
            "VOCAB_INLP_EN":DATA_HOME +"data/" + "en_es/en_vocab_inlp.txt",
            "VOCAB_EN":DATA_HOME +"data/" + "en_es/en_vocab.txt",
        }
}


def parse_config(config_str):
    return ast.literal_eval(config_str)


def get_basic_configurations(config_str):
    config = parse_config(config_str)
    USE_DEBIASED = config["USE_DEBIASED"]
    LANGUAGE = config["LANGUAGE"]
    if "COLLECT_EMBEDDING_TABLE" in config.keys():
        COLLECT_EMBEDDING_TABLE = config["COLLECT_EMBEDDING_TABLE"]
    else:
        COLLECT_EMBEDDING_TABLE = None
    DEBIAS_METHOD = config["DEBIAS_METHOD"]
    TRANSLATION_MODEL = config["TRANSLATION_MODEL"]
    DEBIAS_ENCODER=config["DEBIAS_ENCODER"]
    BEGINNING_DECODER_DEBIAS=config["BEGINNING_DECODER_DEBIAS"]
    END_DECODER_DEBIAS=config["END_DECODER_DEBIAS"]
    WORDS_TO_DEBIAS=config["WORDS_TO_DEBIAS"]
    return USE_DEBIASED, LANGUAGE, COLLECT_EMBEDDING_TABLE, DEBIAS_METHOD, TRANSLATION_MODEL,DEBIAS_ENCODER,BEGINNING_DECODER_DEBIAS,END_DECODER_DEBIAS,WORDS_TO_DEBIAS



def get_debias_files_from_config(config_str):
    config = parse_config(config_str)
    lang = LANGUAGE_STR_MAP[Language(config['LANGUAGE'])]
    debias_method = str(config['DEBIAS_METHOD'])
    translation_model = TranslationModels[int(config['TRANSLATION_MODEL'])]

    DICT_SIZE = param_dict[Language(int(config['LANGUAGE']))]["DICT_SIZE"]

    # the source english dictionary
    ENG_DICT_FILE = param_dict[Language(int(config['LANGUAGE']))]["ENG_DICT_FILE"]

    # the path of the file that translate wrote the embedding table to. this file will be parsed and debiased
    OUTPUT_TRANSLATE_FILE = OUTPUTS_HOME + "en-" + lang + "/debias/output_translate_" + lang + ".txt"

    # the file to which the initial embedding table is pickled to after parsing the file written when running translate
    EMBEDDING_TABLE_FILE = OUTPUTS_HOME + "en-" + lang + "/debias/embedding_table_" + lang + ".bin"

    # the file to which the initial (non debiased) embedding is written in the format of [word] [embedding]\n which is the format debiaswe uses. this is ready to be debiased
    EMBEDDING_DEBIASWE_FILE = OUTPUTS_HOME + "en-" + lang + "/debias/embedding_debiaswe_" + lang +"_"+translation_model+ ".txt"

    # the file to which the debiased embedding table is saved at the end
    DEBIASED_EMBEDDING = OUTPUTS_HOME + "en-" + lang + "/debias/hard-debiased-" + lang + "-" + debias_method +"-"+translation_model+ ".txt"

    now = datetime.now()
    SANITY_CHECK_FILE = OUTPUTS_HOME + "en-" + lang + "/debias/sanity_check_" + now.strftime("%d-%m-%Y_%H-%M-%S") + ".csv"

    DEFINITIONAL_FILE_TARGET_LANG = DATA_HOME+"professions_annotations/" + lang + "_definitional_pairs.json"
    return DICT_SIZE, ENG_DICT_FILE, OUTPUT_TRANSLATE_FILE, EMBEDDING_TABLE_FILE, EMBEDDING_DEBIASWE_FILE, DEBIASED_EMBEDDING, SANITY_CHECK_FILE,DEFINITIONAL_FILE_TARGET_LANG


def get_evaluate_gender_files(config_str):
    config = parse_config(config_str)
    lang = LANGUAGE_STR_MAP[Language(config['LANGUAGE'])]
    debias_method = str(config['DEBIAS_METHOD'])
    translation_model = TranslationModels[int(config['TRANSLATION_MODEL'])]
    DEBIAS_ENCODER = locations_map["DEBIAS_ENCODER"] if config["DEBIAS_ENCODER"] else ""
    BEGINNING_DECODER_DEBIAS = locations_map["BEGINNING_DECODER_DEBIAS"] if config["BEGINNING_DECODER_DEBIAS"] else ""
    END_DECODER_DEBIAS = locations_map["END_DECODER_DEBIAS"] if config["END_DECODER_DEBIAS"] else ""

    # the translations of anti sentences, using the debiased embedding table, with source line nums printed
    ANTI_TRANSLATED_DEBIASED = OUTPUTS_HOME + "en-" + lang + "/output/debiased_anti_" + debias_method +"_"+translation_model+DEBIAS_ENCODER+BEGINNING_DECODER_DEBIAS+END_DECODER_DEBIAS+ ".out.tmp"

    # the translations of anti sentences, using the non debiased embedding table, with source line nums printed
    ANTI_TRANSLATED_NON_DEBIASED = OUTPUTS_HOME + "en-" + lang + "/output/non_debiased_anti_" + debias_method +"_"+translation_model+DEBIAS_ENCODER+BEGINNING_DECODER_DEBIAS+END_DECODER_DEBIAS+".out.tmp"


    # the full anti sentences in english (in the format <gender> <profession location> <sentence> <profession>)
    EN_ANTI_MT_GENDER = MT_GENDER_HOME + "data/aggregates/en_anti.txt"
    # the full sentences in english (in the format <gender> <profession location> <sentence> <profession>)
    EN_NEUTRAL_MT_GENDER = MT_GENDER_HOME + "data/aggregates/en.txt"


    # file prepared to evaluation in the form of source_sentence ||| translated_sentence. translated using debiased embedding table
    DEBIASED_EVAL = MT_GENDER_HOME + "translations/"+translation_model+"/en-" + lang + "-debiased-"+debias_method +"_"+translation_model+DEBIAS_ENCODER+BEGINNING_DECODER_DEBIAS+END_DECODER_DEBIAS+ ".txt"

    # file prepared to evaluation in the form of source_sentence ||| translated_sentence. translated using non debiased embedding table
    NON_DEBIASED_EVAL = MT_GENDER_HOME + "translations/"+translation_model+"//en-" + lang + "-non-debiased-"+debias_method +"_"+translation_model+DEBIAS_ENCODER+BEGINNING_DECODER_DEBIAS+END_DECODER_DEBIAS+".txt"



    return ANTI_TRANSLATED_DEBIASED, ANTI_TRANSLATED_NON_DEBIASED, DEBIASED_EVAL, NON_DEBIASED_EVAL, EN_ANTI_MT_GENDER, EN_NEUTRAL_MT_GENDER


def get_evaluate_translation_files(config_str):
    config = parse_config(config_str)
    lang = LANGUAGE_STR_MAP[Language(config['LANGUAGE'])]
    debias_method = str(config['DEBIAS_METHOD'])
    translation_model = TranslationModels[int(config['TRANSLATION_MODEL'])]
    DEBIAS_ENCODER = locations_map["DEBIAS_ENCODER"] if config["DEBIAS_ENCODER"] else ""
    BEGINNING_DECODER_DEBIAS = locations_map["BEGINNING_DECODER_DEBIAS"] if config["BEGINNING_DECODER_DEBIAS"] else ""
    END_DECODER_DEBIAS = locations_map["END_DECODER_DEBIAS"] if config["END_DECODER_DEBIAS"] else ""
    # data of the gold translation sentences for Bleu evaluation
    BLEU_GOLD_DATA = param_dict[Language(int(config['LANGUAGE']))]["BLEU_GOLD_DATA"]

    # data of the gold translation sentences for Bleu evaluation
    BLEU_GOLD_DATA_NON_TOKENIZED = param_dict[Language(int(config['LANGUAGE']))]["BLEU_GOLD_DATA_NON_TOKENIZED"]

    BLEU_GOLD_DATA_NON_TOKENIZED_EN = param_dict[Language(int(config['LANGUAGE']))]["BLEU_GOLD_DATA_NON_TOKENIZED_EN"]
    # the translations of the dataset sentences, using the debiased embedding table, with source line nums printed
    TRANSLATED_DEBIASED = OUTPUTS_HOME + "en-" + lang + "/output/debiased_" + debias_method + "_"+translation_model+DEBIAS_ENCODER+BEGINNING_DECODER_DEBIAS+END_DECODER_DEBIAS+".out.tmp"

    # the translations of the dataset sentences, using the non debiased embedding table, with source line nums printed
    TRANSLATED_NON_DEBIASED = OUTPUTS_HOME + "en-" + lang + "/output/non_debiased_" + debias_method +"_"+translation_model+ DEBIAS_ENCODER+BEGINNING_DECODER_DEBIAS+END_DECODER_DEBIAS+".out.tmp"

    return BLEU_GOLD_DATA,BLEU_GOLD_DATA_NON_TOKENIZED, BLEU_GOLD_DATA_NON_TOKENIZED_EN, TRANSLATED_DEBIASED, TRANSLATED_NON_DEBIASED


#################debiaswe files#################
DEFINITIONAL_FILE = PROJECT_HOME + "debiaswe/data/definitional_pairs.json"
GENDER_SPECIFIC_FILE = PROJECT_HOME + "debiaswe/data/gender_specific_full.json"
PROFESSIONS_FILE = PROJECT_HOME + "debiaswe/data/professions.json"
EQUALIZE_FILE = PROJECT_HOME + "debiaswe/data/equalize_pairs.json"
