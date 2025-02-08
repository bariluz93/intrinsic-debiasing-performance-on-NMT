#!/bin/bash
set -e
#SBATCH --mem=128g
#SBATCH -c4
#SBATCH --time=7-0
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=bar.iluz@mail.huji.ac.il
#SBATCH --output=/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/slurm/evaluate_gender_bias-%j.out
echo "**************************************** in evaluate_gender_bias.sh ****************************************"

SHORT=l:,d:,t,a,b,e,w:,h
LONG=language:,debias_method:,translate,debias_encoder,beginning_decoder_debias,end_decoder_debias,words_to_debias,help
OPTS=$(getopt -a -n debias --options $SHORT --longoptions $LONG -- "$@")

eval set -- "$OPTS"

translate=false
debias_encoder=0
beginning_decoder_debias=0
end_decoder_debias=0
words_to_debias=""

while :
do
  case "$1" in
    -l | --language )
      language="$2"
      shift 2
      ;;
    -d | --debias_method )
      debias_method="$2"
      shift 2
      ;;
    -t | --translate )
      translate=true
      shift 1
      ;;
    -a | --debias_encoder )
      debias_encoder=1
      shift 1
      ;;
    -b | --beginning_decoder_debias )
      beginning_decoder_debias=1
      shift 1
      ;;
    -e | --end_decoder_debias )
      end_decoder_debias=1
      shift 1
      ;;
    -w | --words_to_debias )
      words_to_debias="$2"
      shift 2
      ;;
    -h | --help)
      echo "usage:
Mandatory arguments:
  -l, --language                  the destination translation language. RUSSIAN = 0, GERMAN = 1,HEBREW = 2,SPANISH = 3 .
  -d, --debias_method             the debias method. HARD_DEBIAS = 0, INLP = 1 .
Optional arguments:
  -p, --preprocess                preprocess the anti dataset .
  -t, --translate                 translate the entire dataset .
  -a, --debias_encoder            debias the encoder .
  -b, --beginning_decoder_debias  debias the decoder inputs .
  -e, --end_decoder_debias        debias the decoder outputs .
  -w, --words_to_debias           set of words to debias. ALL_VOCAB = 0, ONE_TOKEN_PROFESSIONS = 1, ALL_PROFESSIONS = 2 .
  -h, --help                      help message .
if none of debias_encoder, beginning_decoder_debias, end_decoder_debias is selected, debias_encoder is selected defaultly
if words_to_debias is not given, ONE_TOKEN_PROFESSIONS = 1 is selected"
      exit 2
      ;;
    --)
      shift;
      break
      ;;
    *)
      echo "Unexpected option: $1"
      exit 1;;
  esac
done
cur_dir=`pwd`
source ${cur_dir}/../scripts/consts.sh ${language} ${debias_method} 1


debias_loc=""
if [ $debias_encoder = 1 ]; then
  debias_loc="${debias_loc}_A"
fi
if [ $beginning_decoder_debias = 1 ]; then
    debias_loc="${debias_loc}_B"
fi
if [ $end_decoder_debias = 1 ]; then
    debias_loc="${debias_loc}_C"
fi

#################### translate anti sentences to test gender bias ####################
input_path=${snapless_data_dir}/anti_data/anti.en
outputh_path_debiased=${debias_outputs_dir}/${language_dir}/output/debiased_anti_${debias_method}_${model_str}${debias_loc}.out.tmp
outputh_path_non_debiased=${debias_outputs_dir}/${language_dir}/output/non_debiased_anti_${debias_method}_${model_str}${debias_loc}.out.tmp
config_debiased="{'USE_DEBIASED': 1, 'LANGUAGE': ${language_num}, 'DEBIAS_METHOD': ${debias_method}, 'TRANSLATION_MODEL': 1, 'DEBIAS_ENCODER': ${debias_encoder}, 'BEGINNING_DECODER_DEBIAS': ${beginning_decoder_debias}, 'END_DECODER_DEBIAS': ${end_decoder_debias}, 'WORDS_TO_DEBIAS': ${words_to_debias}}"
config_non_debiased="{'USE_DEBIASED': 0, 'LANGUAGE': ${language_num}, 'DEBIAS_METHOD': ${debias_method}, 'TRANSLATION_MODEL': 1, 'DEBIAS_ENCODER': ${debias_encoder}, 'BEGINNING_DECODER_DEBIAS': ${beginning_decoder_debias}, 'END_DECODER_DEBIAS': ${end_decoder_debias}, 'WORDS_TO_DEBIAS': ${words_to_debias}}"

if [ $translate = true ]; then
  echo "#################### translate anti debias ####################"
  echo "python ${debias_files_dir}/src/translate_easynmt.py \
       -i $input_path \
       -o ${outputh_path_debiased} \
       -c ${config_debiased}"
  python ${debias_files_dir}/src/translate_easynmt.py \
       -i "$input_path" \
       -o "${outputh_path_debiased}" \
       -c "${config_debiased}"
  echo "#################### translate anti non debias ####################"
  python ${debias_files_dir}/src/translate_easynmt.py \
       -i "$input_path" \
       -o "${outputh_path_non_debiased}" \
       -c "${config_non_debiased}"
fi


echo "#################### prepare gender data ####################"
python ${debias_files_dir}/src/prepare_gender_data.py -c "${config_non_debiased}"

echo "#################### gender evaluation ####################"

output_result_path=${debias_outputs_dir}/${language_dir}/debias/gender_evaluation_${dst_language}_${debias_method}_${model_str}${debias_loc}.txt
exec > ${output_result_path}
exec 2>&1
cd ${mt_gender_dir}
#source venv/bin/activate
cd src
export FAST_ALIGN_BASE=/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/fast_align
echo "./../scripts/evaluate_language.sh ../data/aggregates/en_anti.txt ${language} ${model_str} ${debias_method} ${debias_loc}"
./../scripts/evaluate_language.sh ../data/aggregates/en_anti.txt ${language} ${model_str} ${debias_method} ${debias_loc}




