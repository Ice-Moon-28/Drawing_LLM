import csv
import os
from generation_pipeline.pipeline import main_pipeline
from generation_pipeline.util import merge_and_sort_csv, read_from_csv, sort_csv_by_id
from openai import OpenAI
from score.score import score
from train_pipeline.pipeline import train

if __name__ == "__main__":
    # main_pipeline(
    #     number=10000,
    #     batch_number=20,
    #     svg_csv='svg10000.csv',
    #     description_csv='description10000.csv',
    #     score_csv='score10000.csv',
    #     start_from_checkoutpoint=False
    # )

    # merge_and_sort_csv(['score10000.csv'], 'score', 'test.csv')
    train()




