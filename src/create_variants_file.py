import json
from consts import DATA_HOME
def create_variants_dict(merged_annotations, lang):
    variants_dict = {}
    professions = set()
    with open(merged_annotations, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        columns = line.split("\t")
        english_profession = columns[0]
        professions.add(english_profession)
        if not english_profession+'-male' in variants_dict:
            variants_dict[english_profession+'-male'] = []
        if not english_profession+'-female' in variants_dict:
            variants_dict[english_profession+'-female'] = []
        for i in range(1, len(columns)):
            if columns[i] != "":
                if i % 2 and columns[i]:
                    variants_dict[english_profession+'-male'].append(columns[i])
                else:
                    variants_dict[english_profession+'-female'].append(columns[i])
    with open(DATA_HOME+"professions_annotations/" + lang + "_variants.json", 'w', encoding='utf-8') as file:
        json.dump(variants_dict, file, ensure_ascii=False)
    return variants_dict, professions


if __name__ == '__main__':
    # de_variants_dict, professions = create_variants_dict(DATA_HOME+"professions_annotations/"+"de_merged_translations_postprocessed.txt","de")
    # he_variants_dict, professions = create_variants_dict(DATA_HOME+"professions_annotations/"+"he_merged_translations_postprocessed.txt","he")
    # ru_variants_dict, professions = create_variants_dict(DATA_HOME+"professions_annotations/"+"ru_merged_translations_postprocessed.txt","ru")
    ru_variants_dict, professions = create_variants_dict(DATA_HOME+"professions_annotations/"+"es_merged_translations_postprocessed.txt","es")