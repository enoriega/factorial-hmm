#!/bin/sh

OUT="results"

for D in *_nocs; do
  for E in "$D"/cp*; do

    echo "Working with $E files..."
    # Link the files
    ln -fs $E/obs.txt .
    ln -fs $E/weights.txt .
    ln -fs $E/states.txt .
    ln -fs $E/test_obs.txt .

    OUT="results/$E"

    if [ ! -d "$OUT" ]; then
      mkdir -p "$OUT"/{hmm,hsmm}
    fi

    echo "Running factorial hmm ..."
    python factorial_run.py
    mv {precision,recall,f1,accuracy,test_log_likelihood,train_log_likelihood}.txt $OUT/hmm
    echo

    echo "Running factorial hsmm ..."
    python factorial_hsmm_run.py
    mv {precision,recall,f1,accuracy,test_log_likelihood,train_log_likelihood}.txt $OUT/hsmm
    echo

  done
done

echo "DONE!!!"
