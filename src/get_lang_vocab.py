import os
import json

from consts import Language, param_dict, DATA_HOME, ENGLISH_VOCAB,LANGUAGE_STR_TO_INT_MAP, get_debias_files_from_config, get_evaluate_gender_files, TRANSFORMERS_CACHE, LANGUAGE_OPPOSITE_STR_MAP,LANGUAGE_CODES_MAP
os.environ['TRANSFORMERS_CACHE'] = TRANSFORMERS_CACHE
from transformers import MarianTokenizer, AutoTokenizer
import re
tokenizer_opus_mt = "Helsinki-NLP/opus-mt-en-"
tokenizer_mbart50 = "facebook/mbart-large-50-one-to-many-mmt"
# tokenizer_mbart50 = "facebook/mbart-large-50-many-to-many-mmt"
# tokenizer_mbart50 = "facebook/m2m100_1.2B"

# config_ru = "{'USE_DEBIASED': 0, 'LANGUAGE': " + str(LANGUAGE_STR_TO_INT_MAP["ru"]) + \
#             ", 'COLLECT_EMBEDDING_TABLE': 0, 'DEBIAS_METHOD': 0, 'TRANSLATION_MODEL': 1, " \
#             "'DEBIAS_ENCODER': 0, 'BEGINNING_DECODER_DEBIAS': 0, 'END_DECODER_DEBIAS': 0, 'WORDS_TO_DEBIAS': 0}"
# config_de = "{'USE_DEBIASED': 0, 'LANGUAGE': " + str(LANGUAGE_STR_TO_INT_MAP["de"]) + \
#             ", 'COLLECT_EMBEDDING_TABLE': 0, 'DEBIAS_METHOD': 0, 'TRANSLATION_MODEL': 1, " \
#             "'DEBIAS_ENCODER': 0, 'BEGINNING_DECODER_DEBIAS': 0, 'END_DECODER_DEBIAS': 0, 'WORDS_TO_DEBIAS': 0}"
# config_he = "{'USE_DEBIASED': 0, 'LANGUAGE': " + str(LANGUAGE_STR_TO_INT_MAP["he"]) + \
#             ", 'COLLECT_EMBEDDING_TABLE': 0, 'DEBIAS_METHOD': 0, 'TRANSLATION_MODEL': 1, " \
#             "'DEBIAS_ENCODER': 0, 'BEGINNING_DECODER_DEBIAS': 0, 'END_DECODER_DEBIAS': 0, 'WORDS_TO_DEBIAS': 0}"
# config_es = "{'USE_DEBIASED': 0, 'LANGUAGE': " + str(LANGUAGE_STR_TO_INT_MAP["es"]) + \
#             ", 'COLLECT_EMBEDDING_TABLE': 0, 'DEBIAS_METHOD': 0, 'TRANSLATION_MODEL': 1, " \
#             "'DEBIAS_ENCODER': 0, 'BEGINNING_DECODER_DEBIAS': 0, 'END_DECODER_DEBIAS': 0, 'WORDS_TO_DEBIAS': 0}"
# _, _, _, _, _, EN_NEUTRAL_MT_GENDER = get_evaluate_gender_files(config_es)

def get_words_from_different_datasets_target_lang(lang):
    words=[]
    with open(DATA_HOME+"professions_annotations/"+lang+"_professions.txt",'r') as f:
        words += f.readlines()
    with open(DATA_HOME+"professions_annotations/"+lang+"_definitional_pairs.json") as f:
        data = json.load(f)
        words += sum(data, [])
    lowers = [w.lower() for w in words]
    with open(DATA_HOME+lang+"_words.txt",'w') as f:
        f.write(" ".join(words+lowers))
        f.write(" ".join(words))
    return DATA_HOME+lang+"_words.txt"

def get_words_from_different_datasets_en(EN_NEUTRAL_MT_GENDER):
    words = []
    with open(DATA_HOME+"inlp_data/vecs.filtered.txt",'r') as f:
        lines = f.readlines()
        words+=[(l.split(" "))[0] for l in lines]

    with open(DATA_HOME+"inlp_data/vecs.gendered.txt",'r') as f:
        lines = f.readlines()
        words+= [(l.split(" "))[0] for l in lines]

    with open(DATA_HOME+"debiaswe/definitional_pairs.json") as f:
        data=json.load(f)
        words+=sum(data, [])
    with open(DATA_HOME+"debiaswe/equalize_pairs.json") as f:
        data=json.load(f)
        words+=sum(data, [])
    with open(DATA_HOME+"debiaswe/gender_specific_full.json") as f:
        data=json.load(f)
        words+=data
    with open(DATA_HOME+"debiaswe/gender_specific_seed.json") as f:
        data=json.load(f)
        words+=data
    with open(DATA_HOME+"debiaswe/gender_specific_seed.json") as f:
        data=json.load(f)
        words+=[i[0] for i in data]
    with open(EN_NEUTRAL_MT_GENDER, 'r') as f:
        lines = f.readlines()
        words += [(line.split("\t")[-1]).strip() for line in lines]
    lowers = [w.lower() for w in words]
    with open(DATA_HOME+"english_words.txt",'w') as f:
        f.write(" ".join(words+lowers))
        f.write(" ".join(words))
    return DATA_HOME+"english_words.txt"

# en_dataset_paths = [DATA_HOME+"data/"+'en_ru_30.11.20/newstest2019-enru.en',
#                     DATA_HOME+"data/"+'en_de_5.8/newstest2012.en',
#                     DATA_HOME+"data/"+'en_he_20.07.21/dev.en',
#                     DATA_HOME+"anti_data/"+'anti.en',
#                     DATA_HOME+"data/"+'en_es/books.en',
#                     get_words_from_different_datasets_en()]
# _, _, _, _, EMBEDDING_DEBIASWE_FILE_RU, _, _,_=get_debias_files_from_config(config_ru)
# _, _, _, _, EMBEDDING_DEBIASWE_FILE_DE, _, _,_=get_debias_files_from_config(config_de)
# _, _, _, _, EMBEDDING_DEBIASWE_FILE_HE, _, _,_=get_debias_files_from_config(config_he)
# _, _, _, _, EMBEDDING_DEBIASWE_FILE_ES, _, _,_=get_debias_files_from_config(config_es)
#
# ru_files = [param_dict[Language.RUSSIAN]["BLEU_GOLD_DATA_NON_TOKENIZED"], get_words_from_differentd_datasets_target_lang('ru')]
# de_files = [param_dict[Language.GERMAN]["BLEU_GOLD_DATA_NON_TOKENIZED"], get_words_from_differentd_datasets_target_lang('de')]
# he_files = [param_dict[Language.HEBREW]["BLEU_GOLD_DATA_NON_TOKENIZED"], get_words_from_differentd_datasets_target_lang('he')]
# es_files = [param_dict[Language.SPANISH]["BLEU_GOLD_DATA_NON_TOKENIZED"]]
# # es_files = [param_dict[Language.SPANISH]["BLEU_GOLD_DATA_NON_TOKENIZED"], get_words_from_differentd_atasets_target_lang('es')]
#
# def get_vocab_heb(tokenizer, src_paths, target_filename):
#     lang_vocab = set()
#     for p in src_paths:
#         with open(p, 'r') as src_file:
#
#             lines = src_file.readlines()
#             with tokenizer.as_target_tokenizer():
#                 for line in lines:
#                     words=re.findall(r'[\w]+', line)
#                     for w in words:
#                         indices = tokenizer(w)['input_ids'][:-1]
#                         tokens = tokenizer.convert_ids_to_tokens(indices)
#                         for i,token in zip(indices,tokens):
#                             if not token[1:].isnumeric() and len(token[1:])>1:
#                                 lang_vocab.add(token + "\t" + str(i))
#     lang_vocab = list(lang_vocab)
#
#     with open(target_filename, 'w+') as dst_vocab:
#         for w in lang_vocab:
#             dst_vocab.write( w + '\n')

# def get_vocab(tokenizer, src_paths, target_filename):
#     lang_vocab = set()
#     for p in src_paths:
#         with open(p, 'r') as src_file:
#             lines = src_file.readlines()
#             with tokenizer.as_target_tokenizer():
#                 for line in lines:
#                     words=re.findall(r'[\w]+', line)
#                     for w in words:
#                         indices = tokenizer(w)['input_ids'][:-1]
#                         if len(indices)==1:
#                             tokens = tokenizer.convert_ids_to_tokens(indices)
#                             if not (tokens[0])[1:].isnumeric() and len((tokens[0])[1:])>1:
#                                 lang_vocab.add(tokens[0]+"\t"+str(indices[0]))
#
#     lang_vocab = list(lang_vocab)
#     with open(target_filename, 'w+') as dst_vocab:
#         for w in lang_vocab:
#             dst_vocab.write( w + '\n')
def get_vocab(tokenizer, src_paths, target_filename):
    lang_vocab = set()
    for p in src_paths:
        with open(p, 'r') as src_file:
            lines = src_file.readlines()
            with tokenizer.as_target_tokenizer():
                for line in lines:
                    indices = tokenizer(line)['input_ids'][:-1]
                    tokens = tokenizer.convert_ids_to_tokens(indices)

                    for i in range(len(tokens)):
                        lang_vocab.add(tokens[i] + "\t" + str(indices[i]))

    lang_vocab = list(lang_vocab)
    with open(target_filename, 'w+') as dst_vocab:
        for w in lang_vocab:
            dst_vocab.write( w + '\n')


def get_en_vocab(tokenizer, src_paths, target_filename):
    tokens_set = set()
    for p in src_paths:
        with open(p, 'r') as src_file:
            lines = [l.strip() for l in src_file.readlines()]
            for line in lines:
                tokens = tokenizer.tokenize(line)
                indices = tokenizer.convert_tokens_to_ids(tokens)
                for i in range(len(tokens)):
                    tokens_set.add(tokens[i] + "\t" + str(indices[i]))
    tokens_set = list(tokens_set)
    with open(target_filename, 'w+') as dst_vocab:
        for w in tokens_set:
            dst_vocab.write( w + '\n')

def save_vocab_inlp_form(general_vocab_filename,language_vocab_filename, target_filename):
    with open(general_vocab_filename,'r') as f1, open(language_vocab_filename,'r') as f2:
        # general_vocab = f1.readlines()
        general_vocab = [l.split(" ") for l in f1]
        embedding_size = general_vocab[0][1]
        if len(general_vocab[0])==2:
            general_vocab = general_vocab[1:]
        # language_vocab = f2.readlines()
        language_vocab = [l.split("\t") for l in f2]
    with open(target_filename,"w+") as f:
        lines_to_write = []
        for w in language_vocab:
            index = int(w[1])
            if w[0] == general_vocab[index][0]:
                lines_to_write.append(w[0]+" "+" ".join(general_vocab[index][1:]))
            else:
                print(w[0] + " was not found in the vocab")
        lines_to_write.insert(0,str(len(lines_to_write))+" "+embedding_size)

        for l in lines_to_write:
            f.write(l)
def get_and_save_all_vocabs(lang,translation_model):
    config = "{'USE_DEBIASED': 0, 'LANGUAGE': " + str(LANGUAGE_STR_TO_INT_MAP[lang]) + \
                ", 'COLLECT_EMBEDDING_TABLE': 0, 'DEBIAS_METHOD': 0, 'TRANSLATION_MODEL': "+str(translation_model)+", " \
                "'DEBIAS_ENCODER': 0, 'BEGINNING_DECODER_DEBIAS': 0, 'END_DECODER_DEBIAS': 0, 'WORDS_TO_DEBIAS': 0}"
    _, _, _, _, _, EN_NEUTRAL_MT_GENDER = get_evaluate_gender_files(config)
    en_dataset_paths = [DATA_HOME +"data/" +'en_ru_30.11.20/newstest2019-enru.en',
                        DATA_HOME +"data/" +'en_de_5.8/newstest2012.en',
                        DATA_HOME +"data/" +'en_he_20.07.21/dev.en',
                        DATA_HOME +"anti_data/" +'anti.en',
                        DATA_HOME +"data/" +'en_es/books.en',
                        get_words_from_different_datasets_en(EN_NEUTRAL_MT_GENDER)]
    _, _, _, _, embedding_debiaswe, _, _,_=get_debias_files_from_config(config)

    files = [param_dict[LANGUAGE_OPPOSITE_STR_MAP[lang]]["BLEU_GOLD_DATA_NON_TOKENIZED"], get_words_from_different_datasets_target_lang(lang)]

    if translation_model == 1:
        tokenizer =MarianTokenizer.from_pretrained(tokenizer_opus_mt+lang)
        get_vocab(tokenizer,files,param_dict[LANGUAGE_OPPOSITE_STR_MAP[lang]]["VOCAB_OPUS_MT"])
        get_en_vocab(tokenizer,en_dataset_paths,param_dict[LANGUAGE_OPPOSITE_STR_MAP[lang]]['VOCAB_EN_OPUS_MT'])
        save_vocab_inlp_form(embedding_debiaswe,param_dict[LANGUAGE_OPPOSITE_STR_MAP[lang]]["VOCAB_OPUS_MT"],param_dict[LANGUAGE_OPPOSITE_STR_MAP[lang]]["VOCAB_INLP_OPUS_MT"])
        save_vocab_inlp_form(embedding_debiaswe,param_dict[LANGUAGE_OPPOSITE_STR_MAP[lang]]['VOCAB_EN_OPUS_MT'],param_dict[LANGUAGE_OPPOSITE_STR_MAP[lang]]["VOCAB_INLP_EN_OPUS_MT"])
    if translation_model == 2:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_mbart50,src_lang="en_XX", tgt_lang=LANGUAGE_CODES_MAP[lang],model_max_length= 1024)
        get_vocab(tokenizer,files,param_dict[LANGUAGE_OPPOSITE_STR_MAP[lang]]["VOCAB_MBART50"])
        get_en_vocab(tokenizer,en_dataset_paths,param_dict[LANGUAGE_OPPOSITE_STR_MAP[lang]]['VOCAB_EN_MBART50'])
        save_vocab_inlp_form(embedding_debiaswe,param_dict[LANGUAGE_OPPOSITE_STR_MAP[lang]]["VOCAB_MBART50"],param_dict[LANGUAGE_OPPOSITE_STR_MAP[lang]]["VOCAB_INLP_MBART50"])
        save_vocab_inlp_form(embedding_debiaswe,param_dict[LANGUAGE_OPPOSITE_STR_MAP[lang]]['VOCAB_EN_MBART50'],param_dict[LANGUAGE_OPPOSITE_STR_MAP[lang]]["VOCAB_INLP_EN_MBART50"])
# if __name__ == '__main__':
    # get_and_save_all_vocabs("de", 2)
#
#
#
#     # get_vocab(tokenizer_ru,ru_files,param_dict[Language.RUSSIAN]["VOCAB"])
#     # get_en_vocab(tokenizer_ru,en_dataset_paths,param_dict[Language.RUSSIAN]['VOCAB_EN'])
#     #
#     # get_vocab(tokenizer_de,de_files,param_dict[Language.GERMAN]["VOCAB"])
#     # get_en_vocab(tokenizer_de,en_dataset_paths,param_dict[Language.GERMAN]['VOCAB_EN'])
#     #
#     # get_vocab(tokenizer_he,he_files,param_dict[Language.HEBREW]["VOCAB"])
#     # get_en_vocab(tokenizer_he,en_dataset_paths,param_dict[Language.HEBREW]['VOCAB_EN'])
#
#     # get_vocab(tokenizer_es,es_files,param_dict[Language.SPANISH]["VOCAB"])
#     # get_en_vocab(tokenizer_es,en_dataset_paths,param_dict[Language.SPANISH]['VOCAB_EN'])
#
#
#     # save_vocab_inlp_form(EMBEDDING_DEBIASWE_FILE_RU,param_dict[Language.RUSSIAN]["VOCAB"],param_dict[Language.RUSSIAN]["VOCAB_INLP"])
#     # save_vocab_inlp_form(EMBEDDING_DEBIASWE_FILE_RU,param_dict[Language.RUSSIAN]['VOCAB_EN'],param_dict[Language.RUSSIAN]["VOCAB_INLP_EN"])
#     #
#     # save_vocab_inlp_form(EMBEDDING_DEBIASWE_FILE_DE,param_dict[Language.GERMAN]["VOCAB"],param_dict[Language.GERMAN]["VOCAB_INLP"])
#     # save_vocab_inlp_form(EMBEDDING_DEBIASWE_FILE_DE,param_dict[Language.GERMAN]['VOCAB_EN'],param_dict[Language.GERMAN]["VOCAB_INLP_EN"])
#     #
#     # save_vocab_inlp_form(EMBEDDING_DEBIASWE_FILE_HE,param_dict[Language.HEBREW]["VOCAB"],param_dict[Language.HEBREW]["VOCAB_INLP"])
#     # save_vocab_inlp_form(EMBEDDING_DEBIASWE_FILE_HE,param_dict[Language.HEBREW]['VOCAB_EN'],param_dict[Language.HEBREW]["VOCAB_INLP_EN"])
#
#     # save_vocab_inlp_form(EMBEDDING_DEBIASWE_FILE_ES,param_dict[Language.SPANISH]["VOCAB"],param_dict[Language.SPANISH]["VOCAB_INLP"])
#     save_vocab_inlp_form(EMBEDDING_DEBIASWE_FILE_ES,param_dict[Language.SPANISH]['VOCAB_EN'],param_dict[Language.SPANISH]["VOCAB_INLP_EN"])