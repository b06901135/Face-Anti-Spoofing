#!/bin/bash

kaggle competitions submit -c dlcv-fall-2020-final-challenge-1-task1 -f output/${1}_siw.csv -m "$2"
kaggle competitions submit -c dlcv-fall-2020-final-challenge-1-task2 -f output/${1}.csv -m "$2"
kaggle competitions submit -c dlcv-fall-2020-final-challenge-1-bonus -f output/${1}_cat.csv -m "$2"
