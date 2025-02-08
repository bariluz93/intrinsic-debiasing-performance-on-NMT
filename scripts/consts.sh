set -e
#module load tensorflow/2.0.0
source /cs/usr/bareluz/gabi_labs/nematus_clean/debias_files/src/venv_test/bin/activate
export snapless_dir=/cs/snapless/gabis/bareluz
#source ${snapless_dir}/nematus_env3/bin/activate
export src_language=en
export dst_language=$1
export debias_method=$2
export model=$3
# set PYTHONPATH
export project_dir=/cs/usr/bareluz/gabi_labs/nematus_clean
export PYTHONPATH=${PYTHONPATH}:${project_dir}/debias_files/src
unset PYTHONHOME
#export PYTHONHOME=${PYTHONHOME}:${snapless_dir}
#export PATH=${PATH}:${project_dir}
#echo "PATH:${PATH}"
# set up parameters
export nematus_dir=/cs/usr/bareluz/gabi_labs/nematus_clean/nematus
export debias_files_dir=/cs/usr/bareluz/gabi_labs/nematus_clean/debias_files
export debias_outputs_dir=/cs/usr/bareluz/gabi_labs/nematus_clean/debias_outputs
export snapless_dir=/cs/snapless/gabis/bareluz
export snapless_data_dir=/cs/snapless/gabis/bareluz/debias_nmt_data
export language_dir=${src_language}-${dst_language}
export mt_gender_dir=/cs/usr/bareluz/gabi_labs/nematus_clean/mt_gender

data_path=/cs/snapless/gabis/bareluz/debias_nmt_data/data

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
	*)
		echo "invalid model given (${model}). the possible models are 0 for Nematus or 1 to easyNMT"
		;;
esac
### done correctness checks

case ${dst_language} in
	ru)
		case ${model_str} in
		NEMATUS)
		  export input_path=${data_path}/${src_language}_${dst_language}_30.11.20/newstest2019-enru.unesc.tok.tc.bpe.en
		  ;;
		EASY_NMT)
      export input_path=${data_path}/${src_language}_${dst_language}_30.11.20/newstest2019-enru.en
      ;;
    esac
		export language_num=0
		;;
	de)
	  case ${model_str} in
		NEMATUS)
		  export input_path=${data_path}/${src_language}_${dst_language}_5.8/newstest2012.unesc.tok.tc.bpe.en
		  ;;
		EASY_NMT)
      export input_path=${data_path}/${src_language}_${dst_language}_5.8/newstest2012.en
      ;;
    esac
		export language_num=1
		;;
	he)
    case ${model_str} in
		NEMATUS)
		  export input_path=${data_path}/${src_language}_${dst_language}_20.07.21/dev.unesc.tok.tc.bpe.en
		  ;;
		EASY_NMT)
      export input_path=${data_path}/${src_language}_${dst_language}_20.07.21/dev.en
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
		EASY_NMT)
      export input_path=${data_path}/${src_language}_${dst_language}/books.en
      ;;
    esac
		export language_num=3
		;;
	*)
		echo "invalid language given. the possible languages are ru, de, he, es"
		;;
esac
