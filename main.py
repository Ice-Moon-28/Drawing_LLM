import csv
import os
from generation_pipeline.pipeline import deepseek_svg_pipeline, read_annotation_to_csv
from generation_pipeline.prompt import generate_svg_prompt
from generation_pipeline.util import read_from_csv, enforce_constraints, extract_svg, read_from_json, save_to_json, save_to_svg, transform_svg_to_png, read_svg_as_string
from openai import OpenAI
from score.score import score

if __name__ == "__main__":
    # from generation_pipeline.pipeline import deepseek_descrption_pipeline
    # deepseek_descrption_pipeline(number=1000, batch_number=10, filename='descriptions.txt')

    read_annotation_to_csv(filename="sorted_clusters.json", output_csv1='svg.csv', output_csv2='description.csv')

    svg = read_from_csv(filename="svg.csv")
    description = read_from_csv(filename="description.csv")

    import pdb; pdb.set_trace()

    svg['svg'] = svg['svg'].astype(str)
    description['description'] = description['description'].astype(str)
    
    score(description, svg, 'id')


    

    # for desc in sorted_descpriptions:
    #     input_svg = f"svg/{desc['label']}.svg"
    #     output_png = f"img/{desc['label']}.png"
    #     description = desc["sentences"]
    #     id = desc["label"]
    #     if not os.path.exists(input_svg):
    #         print(f"未找到 {input_svg}，正在生成...")

    #         client = OpenAI(api_key='sk-8a652d9cc48342789bd2657f9be44c2e', base_url="https://api.deepseek.com")

    #         messages = generate_svg_prompt(description)

    #         response = client.chat.completions.create(
    #             model="deepseek-chat",
    #             messages=messages,
    #             stream=False
    #         )

    #         response = response.choices[0].message.content

    #         response = extract_svg(
    #             response
    #         )

    #         if response is None:
    #             continue

    #         print("🤖 Response: ", response)

    #         clear_response = enforce_constraints(response)

    #         print("🤖 Clear Response: ", clear_response)

    #         if clear_response is None:
    #             continue

    #         save_to_svg(clear_response, filename=f"svg/{id}.svg")
            
    #         transform_svg_to_png(input_svg, output_png)
    #     else:
    #         print(f"{input_svg} 已存在，跳过生成")
