#!/bin/bash
set -e


cur_dir=`pwd`
# the dst language here doesn't matter
source ${cur_dir}/../scripts/consts.sh ru 0 ${model}

echo "#################### cleanup ####################"
python ${debias_files_dir}/src/cleanup.py ${collect_embedding_table} ${translate}

#################### 1 tokens professions ###################


echo run begining encoder
collect_embedding_table=""
preprocess=""
translate="-t"
debias_encoder="-a"
beginning_decoder_debias=""
end_decoder_debias=""
words_to_debias="1"
model="1"

echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ es 0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
sh ${debias_files_dir}/scripts/run_all_flows.sh -l es -d 0 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ es 1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
sh ${debias_files_dir}/scripts/run_all_flows.sh -l es -d 1 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}

echo run begining decoder

collect_embedding_table=""
preprocess=""
translate="-t"
debias_encoder=""
beginning_decoder_debias="-b"
end_decoder_debias=""
words_to_debias="1"
model="1"

echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ es 0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
sh ${debias_files_dir}/scripts/run_all_flows.sh -l es -d 0 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ es 1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
sh ${debias_files_dir}/scripts/run_all_flows.sh -l es -d 1 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}

echo run end decoder

collect_embedding_table=""
preprocess=""
translate="-t"
debias_encoder=""
beginning_decoder_debias=""
end_decoder_debias="-e"
words_to_debias="1"
model="1"

echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ es 0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
sh ${debias_files_dir}/scripts/run_all_flows.sh -l es -d 0 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ es 1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
sh ${debias_files_dir}/scripts/run_all_flows.sh -l es -d 1 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}

#echo run begining encoder + beginning decoder
#
#collect_embedding_table=""
#preprocess=""
#translate="-t"
#debias_encoder="-a"
#beginning_decoder_debias="-b"
#end_decoder_debias=""
#words_to_debias="1"
#model="1"
#
#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ es 0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l es -d 0 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ es 1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l es -d 1 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}

#echo run begining encoder + end decoder
#
#collect_embedding_table=""
#preprocess=""
#translate="-t"
#debias_encoder="-a"
#beginning_decoder_debias=""
#end_decoder_debias="-e"
#words_to_debias="1"
#model="1"
#
#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ es 0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l es -d 0 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ es 1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l es -d 1 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
#echo run begining decoder + end decoder
#
#collect_embedding_table=""
#preprocess=""
#translate="-t"
#debias_encoder=""
#beginning_decoder_debias="-b"
#end_decoder_debias="-e"
#words_to_debias="1"
#model="1"
#
#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ es 0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l es -d 0 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ es 1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l es -d 1 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
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
#model="1"
#
#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ es 0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l es -d 0 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
#echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ es 1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#sh ${debias_files_dir}/scripts/run_all_flows.sh -l es -d 1 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}



