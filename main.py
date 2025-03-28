import csv
import os
from generation_pipeline.pipeline import main_pipeline
from generation_pipeline.util import sort_csv_by_id
from openai import OpenAI
from score.score import score
from train_pipeline.train import train

if __name__ == "__main__":
    main_pipeline(
        number=100,
        batch_number=20,
        svg_csv='svg1.csv',
        description_csv='des1.csv',
        score_csv='score20001.csv',
        start_from_checkoutpoint=False
    )

    # train()

    # import pandas as pd
    # import csv

    # # 读取原始CSV
    # df = pd.read_csv('test.csv')

    # # 拆分为两个DataFrame
    # df_id_svg = df[['id', 'svg']]
    # df_id_description = df[['id', 'description']]

    # # 保存为两个新CSV文件
    # df_id_svg.to_csv('id_svg.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)
    # df_id_description.to_csv('id_description.csv', index=False)