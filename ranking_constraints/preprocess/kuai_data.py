import random

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

#from config import dataConfig as dc
from ranking_constraints.config import create_parser, parse_args, parse_dataset_args

if __name__ == "__main__":
    #dc.init_config("kuai")
    configs = create_parser()
    configs = parse_dataset_args(configs)
    #DATA_DIR, EXP_DIR = dc.DATA_DIR, dc.EXP_DIR
    DATA_DIR, EXP_DIR = configs['DATA_DIR'], configs['EXP_DIR']

    random.seed(0)

    matrix_path = DATA_DIR / "data/small_matrix.csv"
    logger.info("Loading small matrix...")
    small_matrix = pd.read_csv(matrix_path)

    users = small_matrix["user_id"].unique().tolist()
    num_user = len(users)
    user_per_video = small_matrix.groupby("video_id")["user_id"].count()
    filtered_videos = user_per_video[user_per_video == num_user].index.tolist()
    filtered_matrix = small_matrix.loc[
        small_matrix["video_id"].isin(filtered_videos)
    ].set_index(["user_id", "video_id"])
    relevance = (filtered_matrix["watch_ratio"] / 2).clip(upper=1).rename("relevance")
    num_video = len(filtered_videos)
    logger.info(f"#User: {num_user}, #Video: {num_video}")

    Z = np.zeros((num_user, num_video))
    for i, user_id in enumerate(users):
        for j, video_id in enumerate(filtered_videos):
            Z[i, j] = relevance.loc[(user_id, video_id)]

    A, R = train_test_split(Z, test_size=0.2, random_state=0)
    D, V = train_test_split(A, test_size=0.25, random_state=0)

    np.save("train.npy", D)
    np.save("dev.npy", V)
    np.save("test.npy", R)
    print(D.shape, V.shape, R.shape)
