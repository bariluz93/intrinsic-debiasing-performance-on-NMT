import os
os.environ['TRANSFORMERS_CACHE'] = '/cs/snapless/gabis/bareluz'
from consts import Language, param_dict,DATA_HOME, MT_GENDER_HOME

from transformers import MarianTokenizer

tokenizer_de = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
tokenizer_he = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-he")
tokenizer_ru = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
tokenizer_es = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")

def get_all_professions():
    professions = set()
    with open(MT_GENDER_HOME + "data/aggregates/en.txt", 'r') as f:
        lines = f.readlines()
    for line in lines:
        professions.add((line.split("\t")[-1]).strip())
    return list(professions)

def tokenize_professions(tokenizer,professions, debias_target_language):
    tokenized_professions = []
    # if we debias the decoder we use decoder's tokenizer
    if debias_target_language:
        with tokenizer.as_target_tokenizer():
            for p in professions:
                indices = tokenizer(p)['input_ids'][:-1]
                t = tokenizer.convert_ids_to_tokens(indices)
                if len(t) == 1:
                    tokenized_professions.append(p)
    else:
        for p in professions:
            t = tokenizer.tokenize(p)
            if '▁' in t:
                t.remove('▁')
            if len(t) == 1:
                tokenized_professions.append(p)

    return tokenized_professions

def get_statistics_one_token_professions(all_professions,one_token_professions,dataset):
    txt = open(dataset).read()
    counts={}
    for p in all_professions:
        counts[p]=txt.count(p)
    return sum([counts[p] for p in one_token_professions])/sum([counts[p] for p in all_professions])


if __name__ == '__main__':
    # all_professions = get_all_professions()
    # with open(DATA_HOME + "professions_annotations/" + "he_professions.txt") as f:
    #     all_hebrew_professions = [p.strip() for p in f.readlines()]
    # with open(DATA_HOME + "professions_annotations/" + "de_professions.txt") as f:
    #     all_german_professions = [p.strip() for p in f.readlines()]
    # with open(DATA_HOME+"professions_annotations/"+"ru_professions.txt") as f:
    #     all_russian_professions = [p.strip() for p in f.readlines()]
    #
    #
    # hebrew_professions = tokenize_professions(tokenizer_he, all_hebrew_professions,debias_target_language=True)
    # print("Hebrew statistics one token professions = "+
    #       str(get_statistics_one_token_professions(all_hebrew_professions,
    #                                                hebrew_professions,
    #                                                param_dict[Language.HEBREW]["BLEU_GOLD_DATA_NON_TOKENIZED"])))
    # professions_en_he = tokenize_professions(tokenizer_he, all_professions, debias_target_language=False)
    # print("English Hebrew statistics one token professions = "+
    #       str(get_statistics_one_token_professions(all_professions,
    #                                                professions_en_he,
    #                                                param_dict[Language.HEBREW]["BLEU_GOLD_DATA_NON_TOKENIZED_EN"])))
    #
    #
    #
    # german_professions = tokenize_professions(tokenizer_de, all_german_professions,debias_target_language=True)
    # print("German statistics one token professions = "+
    #       str(get_statistics_one_token_professions(all_german_professions,
    #                                                german_professions,
    #                                                param_dict[Language.GERMAN]["BLEU_GOLD_DATA_NON_TOKENIZED"])))
    # professions_en_de = tokenize_professions(tokenizer_de, all_professions, debias_target_language=False)
    # print("English German statistics one token professions = "+
    #       str(get_statistics_one_token_professions(all_professions,
    #                                                professions_en_de,
    #                                                param_dict[Language.HEBREW]["BLEU_GOLD_DATA_NON_TOKENIZED_EN"])))
    #
    #
    #
    #
    # russian_professions = tokenize_professions(tokenizer_ru, all_russian_professions,debias_target_language=True)
    # print("Russian statistics one token professions = " +
    #       str(get_statistics_one_token_professions(all_russian_professions,
    #                                                russian_professions,
    #                                                param_dict[Language.RUSSIAN]["BLEU_GOLD_DATA_NON_TOKENIZED"])))
    # professions_en_ru = tokenize_professions(tokenizer_ru, all_professions, debias_target_language=False)
    # print("English Russian statistics one token professions = " +
    #       str(get_statistics_one_token_professions(all_professions,
    #                                                professions_en_ru,
    #                                                param_dict[Language.RUSSIAN]["BLEU_GOLD_DATA_NON_TOKENIZED_EN"])))
    import json
    with open('/cs/snapless/gabis/bareluz/debias_nmt_data/professions_annotations/es_definitional_pairs.json','rb') as f:
        defs=json.load(f)
