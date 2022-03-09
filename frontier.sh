#!/bin/bash
DATA_HOME="./data"
DATA_FILENAME="./data.pt"

# uncomment this line if you changed `pool_size'
# rm ${DATA_FILENAME}

# set your arguments here
python frontier.py                    \
    --data_home ${DATA_HOME}          \
    --data_year 2021                  \
    --data_filename ${DATA_FILENAME}  \
    --pool_size 10                    \
    --short_selling 1                 \
    --min_return -0.0005              \
    --max_return 0.001                \
    --num_plot_pts 15                 \
    --save_fig_name "./frontier.png"  \
    --save_fig_dpi 300                \
    --save_pts_name "./frontier.csv"