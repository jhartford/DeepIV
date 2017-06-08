SCRIPT_DIR=$(dirname "$0")
export RESULTS=/var/storage/shared/msrlabs/t-jahar/DeepIV/experiments/nonpar.csv
cd /var/storage/shared/msrlabs/t-jahar/DeepIV/experiments/
echo "STARTING JOB 1"
touch starting_job1.txt
python nonpar.py -n 1000 -s 9 --endo 0.5 --results $RESULTS &
python nonpar.py -n 1000 -s 10 --endo 0.5 --results $RESULTS &
python nonpar.py -n 1000 -s 11 --endo 0.5 --results $RESULTS &
python nonpar.py -n 1000 -s 12 --endo 0.5 --results $RESULTS &
python nonpar.py -n 1000 -s 13 --endo 0.5 --results $RESULTS &
python nonpar.py -n 1000 -s 14 --endo 0.5 --results $RESULTS &
python nonpar.py -n 1000 -s 15 --endo 0.5 --results $RESULTS &
python nonpar.py -n 1000 -s 16 --endo 0.5 --results $RESULTS --heartbeat 
exit 0
