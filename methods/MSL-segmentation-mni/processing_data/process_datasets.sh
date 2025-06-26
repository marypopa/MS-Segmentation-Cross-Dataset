# ./process_datasets.sh "quantile"
N=$1
python preparing_3D_MR_MS_input_data.py -n $N
python preparing_isbi_input_data.py -n $N
python preparing_msseg_2016_input_data.py -n $N


