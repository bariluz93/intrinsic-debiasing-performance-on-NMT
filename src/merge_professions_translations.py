from consts import DATA_HOME



hebrew_file_names = [DATA_HOME+"professions_annotations/"+"he1_translations", DATA_HOME+"professions_annotations/"+"he2_translations",
                     DATA_HOME+"professions_annotations/"+"he3_translations"]
german_file_names = [DATA_HOME+"professions_annotations/"+"de1_translations", DATA_HOME+"professions_annotations/"+"de2_translations",
                     DATA_HOME+"professions_annotations/"+"de3_translations"]
russian_file_names = [DATA_HOME+"professions_annotations/"+"ru1_translations", DATA_HOME+"professions_annotations/"+"ru2_translations",
                     DATA_HOME+"professions_annotations/"+"ru3_translations"]
spanish_file_names = [DATA_HOME+"professions_annotations/"+"es1_translations", DATA_HOME+"professions_annotations/"+"es2_translations",
                     DATA_HOME+"professions_annotations/"+"es3_translations"]


def merge_translations(file_names, target_file):
    translations_dict = {}
    professions = set()
    for file in file_names:
        with open(file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            columns = line.split("\t")
            english_profession = columns[0]
            professions.add(english_profession)
            if not english_profession in translations_dict:
                translations_dict[english_profession] = {'Male': set(), 'Female': set()}
            for i in range(1, len(columns)):
                if columns[i] != "":
                    if i % 2 and columns[i]:
                        translations_dict[english_profession]['Male'].add(columns[i])
                    else:
                        translations_dict[english_profession]['Female'].add(columns[i])
    with open(target_file, 'w+') as f:
        f.write(str(translations_dict))
    return translations_dict, professions

def get_professions_list(translations_dict, professions, target_file):
    target_lang_professions = set()
    for p in professions:
        target_lang_professions.update(translations_dict[p]['Male'])
        target_lang_professions.update(translations_dict[p]['Female'])
    with open(target_file,'w') as f:
        for p in target_lang_professions:
            f.write(p+"\n")
    return target_lang_professions





if __name__ == '__main__':
    # this was in the case where we had all annotations and we wanted a sum of them
    # he_translations, professions = merge_translations(hebrew_file_names, DATA_HOME+"professions_annotations/"+"he_merged_translations.txt")
    # de_translations, professions = merge_translations(german_file_names, DATA_HOME+"professions_annotations/"+"de_merged_translations.txt")
    # ru_translations, professions = merge_translations(russian_file_names, DATA_HOME+"professions_annotations/"+"ru_merged_translations.txt")
    es_translations, professions = merge_translations(spanish_file_names, DATA_HOME+"professions_annotations/"+"es_merged_translations.txt")

    # he_professions = get_professions_list(he_translations, professions,DATA_HOME+"professions_annotations/"+"he_professions.txt")
    # de_professions = get_professions_list(de_translations, professions,DATA_HOME+"professions_annotations/"+"de_professions.txt")
    # ru_professions = get_professions_list(ru_translations, professions,DATA_HOME+"professions_annotations/"+"ru_professions.txt")
    es_professions = get_professions_list(es_translations, professions,DATA_HOME+"professions_annotations/"+"es_professions.txt")

