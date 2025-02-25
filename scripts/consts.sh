#!/bin/bash
set -e

# change the following dirs to your dir paths

# working dir contains project dir, and the virtual environment.
export working_dir=/cs/usr/bareluz/gabi_labs
# project dir is the directory where all repositories are cloned into
# (debias_file, mt_gender, debiaswe, EasyNmt, nematus, nullspace_projection, transformers)
export project_dir=${working_dir}/nematus_clean
# snapless_data_dir is the dir where the data and models are saved at
export snapless_data_dir=/cs/snapless/gabis/bareluz/debias_nmt_data

export FAST_ALIGN_BASE=${project_dir}/nematus/fast_align
source ${working_dir}/miniconda3/etc/profile.d/conda.sh
conda activate ${working_dir}/miniconda3/envs/conda_env
#source ${project_dir}/venv/bin/activate
# TODO unite two environments
export src_language=en
export dst_language=$1
export debias_method=$2
export model=$3
export PYTHONPATH=${PYTHONPATH}:${project_dir}/debias_files/src
unset PYTHONHOME
# set up parameters
export nematus_dir=${project_dir}/nematus
export debias_files_dir=${project_dir}/debias_files
export debias_outputs_dir=${project_dir}/debias_outputs
export language_dir=${src_language}-${dst_language}
export mt_gender_dir=${project_dir}/mt_gender
#ru_data = DATA_HOME + "en_ru_30.11.20/newstest2019-enru.en"
#de_data = DATA_HOME + "en_de_5.8/newstest2012.en"
#he_data = DATA_HOME + "en_he_20.07.21/dev.en"

### correctness checks

case ${model} in
	0)
		export model_str=NEMATUS
		;;
	1)
		export model_str=EASY_NMT
		;;
  2)
    export model_str=MBART50
    ;;
	*)
		echo "invalid model given (${model}). the possible models are 0 for Nematus or 1 to easyNMT or 2 to MBART50"
		;;
esac
### done correctness checks

case ${dst_language} in
	ru)
		case ${model_str} in
		NEMATUS)
		  export input_path=${snapless_data_dir}/data/${src_language}_${dst_language}_30.11.20/newstest2019-enru.unesc.tok.tc.bpe.en
		  ;;
		EASY_NMT|MBART50)
      export input_path=${snapless_data_dir}/data/${src_language}_${dst_language}_30.11.20/newstest2019-enru.en
      ;;
    esac
		export language_num=0
		;;
	de)
	  case ${model_str} in
		NEMATUS)
		  export input_path=${snapless_data_dir}/data/${src_language}_${dst_language}_5.8/newstest2012.unesc.tok.tc.bpe.en
		  ;;
		EASY_NMT|MBART50)
      export input_path=${snapless_data_dir}/data/${src_language}_${dst_language}_5.8/newstest2012.en
      ;;
    esac
		export language_num=1
		;;
	he)
    case ${model_str} in
		NEMATUS)
		  export input_path=${snapless_data_dir}/data/${src_language}_${dst_language}_20.07.21/dev.unesc.tok.tc.bpe.en
		  ;;
		EASY_NMT|MBART50)
      export input_path=${snapless_data_dir}/data/${src_language}_${dst_language}_20.07.21/dev.en
      ;;
    esac
		export language_num=2
		;;
  es)
    case ${model_str} in
		NEMATUS)
		  echo "NEMATUS doesn't support spanish translation"
		  exit 1
		  ;;
		EASY_NMT|MBART50)
      export input_path=${snapless_data_dir}/data/${src_language}_${dst_language}/books.en
      ;;
    esac
		export language_num=3
		;;
	*)
		echo "invalid language given. the possible languages are ru, de, he, es"
		;;
esac
