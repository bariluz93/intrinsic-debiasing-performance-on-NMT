#!/bin/bash
set -e
pairs_file=$1
category=$2
score_type=$3

case ${score_type} in
	gender_scores)
		statistical_test="McNemar"
		;;
	blue_scores)
		statistical_test="t-test"
		;;
	*)
		echo "invalid score_type given (${score_type}). the possible models are gender_scores or blue_scores"
		;;
esac
statistical_test_dir="/cs/usr/bareluz/gabi_labs/nematus_clean/debias_outputs/results_for_statistic_significance/"
readarray -t pairs < ${pairs_file}
output_result_path=${statistical_test_dir}${score_type}_results_${category}.txt
exec > ${output_result_path}
exec 2>&1
for pair in "${pairs[@]}"
do
  IFS=',' read -ra p <<< $pair
  echo "****** ${p[1]} --- ${p[0]} ******"
  echo ${statistical_test} | python ../../testSignificanceNLP/testSignificance.py ${p[0]} ${p[1]} 0.05
done

#translation_output_result_path=${statistical_test_dir}translation_results_${category}.txt
#exec > ${translation_output_result_path}
#exec 2>&1
#translation_results_dir=${statistical_test_dir}ready_for_statistical_significance_test/"blue_scores/"
#
#for pair in "${pairs[@]}"
#do
#  IFS=',' read -ra p <<< $pair
#  echo "****** ${p[1]} --- ${p[0]} ******"
#  echo Permutation | python ../../testSignificanceNLP/testSignificance.py ${p[0]} ${p[1]} 0.05
#done
#LANGUAGES=("he" "de" "ru")
#LOCATIONS=("A" "B" "C")
#DEBIAS_METHODS=("0" "1")
#for lang in "${LANGUAGES[@]}"
#do
#  for m in "${DEBIAS_METHODS[@]}"
#  do
#    for loc in "${LOCATIONS[@]}"
#    do
#      filename_non_debiased=${translation_results_dir}${lang}/non_debiased_${m}_EASY_NMT_${loc}.out.txt
#      filename_debiased=${translation_results_dir}${lang}/debiased_${m}_EASY_NMT_${loc}.out.txt
#      echo "****** ${filename_non_debiased} --- ${filename_debiased} ******"
##      echo "Permutation | python ../../testSignificanceNLP/testSignificance.py ${filename_non_debiased} ${filename_debiased} 0.05"
#      echo  Permutation | python ../../testSignificanceNLP/testSignificance.py ${filename_non_debiased} ${filename_debiased} 0.05
#    done
#  done
#done
