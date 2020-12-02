#!/bin/bash

echo "##########################################################################"
echo "###                         MuZero RL agent                            ###"
echo "##########################################################################"

declare -i nb_epochs=15
declare -i nb_episodes=1000
declare -i nb_opti_steps=500
declare -i nb_leaves_per_move=50
declare -i nb_games=100
declare -i rd_seed=42

echo "  Step #1 - launch program with fixed parameters"
python3 main.py $nb_epochs $nb_episodes $nb_opti_steps $nb_leaves_per_move $nb_games $rd_seed
