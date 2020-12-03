#!/bin/bash

echo "##########################################################################"
echo "###                         MuZero RL agent                            ###"
echo "##########################################################################"

# print('"{}" "{}" "{}"'.format(*my_python_function())
declare -i nb_epochs=15
declare -i nb_episodes=1000
declare -i nb_opti_steps=500
declare -i nb_leaves_per_move=50
declare -i nb_games=100
declare -i rd_seed=42

mkdir tmp
touch tmp/white_results.dat
touch tmp/black_results.dat

echo "  Step #1 - launch program with fixed parameters"
python3 main.py $nb_epochs $nb_episodes $nb_opti_steps $nb_leaves_per_move $nb_games $rd_seed

echo "  Step #2 - Trying different combinations of hyperparameters"
for epochs in `awk 'BEGIN { for( i=1; i<=20; i+=2 ) print i }'`;
do
    for episodes in `awk 'BEGIN { for( j=100; j<=10000; j*=2 ) print j }'`;
    do
        for episodes in `awk 'BEGIN { for( j=100; j<=10000; j*=2 ) print j }'`;
        do
            for opti_steps in `awk 'BEGIN { for( k=10; k<=100000; k*=5 ) print k }'`;
            do
                white_score=0
                black_score=0
                nb_seeds=0
                for seed in `awk 'BEGIN { for( l=10; l<=1000; l*=3 ) print l }'`;
                do
                    echo "Running algorithm for $epochs epochs, $episodes episodes and $opti_steps opti_steps..."
                    touch /tmp/black_$epochs_$episodes_$opti_steps_$seed.txt
                    touch /tmp/white_$epochs_$episodes_$opti_steps_$seed.txt
                    tmp=`python3 main.py $epochs $episodes $opti_steps $nb_leaves_per_move $nb_games $seed`
                    tmp_white = tmp[0]
                    tmp_black = tmp[1]
                    white_score+=tmp_white
                    black_score+=tmp_black
                    nb_seeds+=1
                done
                avg_white=white_score/nb_seeds
                avg_black=black_score/nb_seeds
                echo -n "$epochs $episodes $avg_white" >> white_results.dat
                echo -n "$epochs $episodes $avg_black" >> black_results.dat
            done
        done
    done
done

echo "Data generated, done"
echo ""

