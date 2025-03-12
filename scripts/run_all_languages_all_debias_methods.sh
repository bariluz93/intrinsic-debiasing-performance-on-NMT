collect_embedding_table=$1
preprocess=$2
translate=$3
debias_encoder=$4
beginning_decoder_debias=$5
end_decoder_debias=$6
words_to_debias=$7
model=$8
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ de 0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
sh ${debias_files_dir}/scripts/run_all_flows.sh -l de -d 0 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ de 1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
sh ${debias_files_dir}/scripts/run_all_flows.sh -l de -d 1 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ he 0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
sh ${debias_files_dir}/scripts/run_all_flows.sh -l he -d 0 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ he 1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
sh ${debias_files_dir}/scripts/run_all_flows.sh -l he -d 1 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ru 0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
sh ${debias_files_dir}/scripts/run_all_flows.sh -l ru -d 0 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ru 1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
sh ${debias_files_dir}/scripts/run_all_flows.sh -l ru -d 1 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}