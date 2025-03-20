from generation_pipeline.pipeline import deepseek_svg_pipeline
from generation_pipeline.util import read_from_json


if __name__ == "__main__":
    # from generation_pipeline.pipeline import deepseek_descrption_pipeline
    # deepseek_descrption_pipeline(number=1000, batch_number=10, filename='descriptions.txt')

    descpriptions = read_from_json(filename="clusters.json")
    print(descpriptions)

    datas = [
        {
            "label": item["label"],
            "sentences": item["sentences"][0]
        }
        for item in descpriptions
    ]

    deepseek_svg_pipeline(
        descrptions=datas
    )