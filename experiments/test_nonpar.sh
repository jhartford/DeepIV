SCRIPT_DIR=$(dirname "$0")
export RESULTS=/var/storage/shared/msrlabs/t-jahar/DeepIV/experiments/nonpar.csv
cd /var/storage/shared/msrlabs/t-jahar/DeepIV/experiments/
echo "STARTING JOB"
touch starting_job.txt
python nonpar.py -n 1000 -s 1 --endo 0.5 --results $RESULTS &
python nonpar.py -n 1000 -s 2 --endo 0.5 --results $RESULTS &
python nonpar.py -n 1000 -s 3 --endo 0.5 --results $RESULTS &
python nonpar.py -n 1000 -s 4 --endo 0.5 --results $RESULTS &
python nonpar.py -n 1000 -s 5 --endo 0.5 --results $RESULTS &
python nonpar.py -n 1000 -s 6 --endo 0.5 --results $RESULTS &
python nonpar.py -n 1000 -s 7 --endo 0.5 --results $RESULTS &
python nonpar.py -n 1000 -s 8 --endo 0.5 --results $RESULTS --heartbeat 
exit 0
