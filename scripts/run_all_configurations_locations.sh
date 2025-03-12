#!/bin/bash
#SBATCH --mem=64g
#SBATCH --time=8:0:0
#SBATCH --gres=gpu:1,vmem:20g
set -e


cur_dir=`pwd`
# the dst language here doesn't matter
source ${cur_dir}/../scripts/consts.sh ru 0 0

################ 1 tokens professions ###################
echo "#################### cleanup ####################"
python ${debias_files_dir}/src/cleanup.py ${collect_embedding_table} ${translate}

echo "run begining encoder"
collect_embedding_table=""
preprocess=""
translate="-t"
debias_encoder="-a"
beginning_decoder_debias=""
end_decoder_debias=""
words_to_debias="1"
model="2"

source ${cur_dir}/../scripts/consts.sh ru 0 ${model}

#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ de 0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l de -d 0 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ de 1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
sh ${debias_files_dir}/scripts/run_all_flows.sh -l de -d 1 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}

#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ de 2 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l de -d 2 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
#
#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ he 0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l he -d 0 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ he 1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
sh ${debias_files_dir}/scripts/run_all_flows.sh -l he -d 1 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}

#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ he 2 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
##sh ${debias_files_dir}/scripts/run_all_flows.sh -l he -d 2 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l he -d 2 -m ${model} ${collect_embedding_table} ${preprocess} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}

#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ru 0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l ru -d 0 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ru 1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
sh ${debias_files_dir}/scripts/run_all_flows.sh -l ru -d 1 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
#
# echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ru 2 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l ru -d 2 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}

#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ es 0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l es -d 0 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ es 1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l es -d 1 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}

echo "run begining decoder"

collect_embedding_table=""
preprocess=""
translate="-t"
debias_encoder=""
beginning_decoder_debias="-b"
end_decoder_debias=""
words_to_debias="1"
model="2"
source ${cur_dir}/../scripts/consts.sh ru 0 ${model}

#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ de 0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l de -d 0 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ de 1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
sh ${debias_files_dir}/scripts/run_all_flows.sh -l de -d 1 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
#
#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ de 2 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l de -d 2 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
#
#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ he 0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l he -d 0 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ he 1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
sh ${debias_files_dir}/scripts/run_all_flows.sh -l he -d 1 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}

#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ he 2 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
##sh ${debias_files_dir}/scripts/run_all_flows.sh -l he -d 2 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l he -d 2 -m ${model} ${collect_embedding_table} ${preprocess} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}

#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ru 0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l ru -d 0 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ru 1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l ru -d 1 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}

echo "run end decoder"

collect_embedding_table=""
preprocess=""
translate="-t"
debias_encoder=""
beginning_decoder_debias=""
end_decoder_debias="-e"
words_to_debias="1"
model="2"
source ${cur_dir}/../scripts/consts.sh ru 0 ${model}

#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ de 0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l de -d 0 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ de 1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
sh ${debias_files_dir}/scripts/run_all_flows.sh -l de -d 1 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
#
#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ de 2 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l de -d 2 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
#
#
#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ he 0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l he -d 0 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ he 1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
sh ${debias_files_dir}/scripts/run_all_flows.sh -l he -d 1 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}

#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ he 2 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
##sh ${debias_files_dir}/scripts/run_all_flows.sh -l he -d 2 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l he -d 2 -m ${model} ${collect_embedding_table} ${preprocess} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}

#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ru 0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l ru -d 0 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ru 1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l ru -d 1 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}


#echo run begining encoder + beginning decoder
#
#collect_embedding_table=""
#preprocess=""
#translate="-t"
#debias_encoder="-a"
#beginning_decoder_debias="-b"
#end_decoder_debias=""
#words_to_debias="1"
#model="2"
#source ${scripts_dir}/consts.sh ru 0 ${model}
#
#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ de 0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#echo "run_all_flows.sh -l de -d 0 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l de -d 0 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ de 1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l de -d 1 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ he 0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l he -d 0 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ he 1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l he -d 1 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ru 0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l ru -d 0 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ru 1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l ru -d 1 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
#
#echo run begining encoder + end decoder
#
#collect_embedding_table=""
#preprocess=""
#translate="-t"
#debias_encoder="-a"
#beginning_decoder_debias=""
#end_decoder_debias="-e"
#words_to_debias="1"
#model="2"
#source ${scripts_dir}/consts.sh ru 0 ${model}
#
#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ de 0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#echo "run_all_flows.sh -l de -d 0 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l de -d 0 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ de 1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l de -d 1 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ he 0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l he -d 0 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ he 1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l he -d 1 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ru 0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l ru -d 0 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ru 1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l ru -d 1 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
#
#echo run begining decoder + end decoder
#
#collect_embedding_table=""
#preprocess=""
#translate="-t"
#debias_encoder=""
#beginning_decoder_debias="-b"
#end_decoder_debias="-e"
#words_to_debias="1"
#model="2"
#source ${scripts_dir}/consts.sh ru 0 ${model}
#
#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ de 0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#echo "run_all_flows.sh -l de -d 0 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l de -d 0 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ de 1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l de -d 1 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ he 0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l he -d 0 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ he 1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l he -d 1 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ru 0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l ru -d 0 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ru 1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l ru -d 1 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
#
#echo run begining encoder + begining decoder + end decoder
#
#collect_embedding_table=""
#preprocess=""
#translate="-t"
#debias_encoder="-a"
#beginning_decoder_debias="-b"
#end_decoder_debias="-e"
#words_to_debias="1"
#model="2"
#source ${scripts_dir}/consts.sh ru 0 ${model}
#
#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ de 0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#echo "run_all_flows.sh -l de -d 0 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l de -d 0 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ de 1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l de -d 1 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ he 0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l he -d 0 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ he 1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l he -d 1 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ru 0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l ru -d 0 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
