import csv
import os
from generation_pipeline.pipeline import main_pipeline
from openai import OpenAI
from score.score import score

if __name__ == "__main__":
    main_pipeline(
        number=100,
        batch_number=10,
        svg_csv='svg.csv',
        description_csv='description.csv',
        score_csv='score.csv'
    )