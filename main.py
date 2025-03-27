import csv
import os
from generation_pipeline.pipeline import main_pipeline
from generation_pipeline.util import sort_csv_by_id
from openai import OpenAI
from score.score import score

if __name__ == "__main__":
    main_pipeline(
        number=2000,
        batch_number=20,
        svg_csv='svg2000.csv',
        description_csv='description2000.csv',
        score_csv='score2000.csv',
        start_from_checkoutpoint=True
    )