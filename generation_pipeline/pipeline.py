import csv
import os

import concurrent
from generation_pipeline.prompt import generate_descrption_prompt, generate_svg_prompt
from generation_pipeline.util import default_svg, enforce_constraints, extract_answers, prompt_with_deepseek, extract_svg, read_from_csv, read_from_json, read_svg_as_string, save_to_json, save_to_svg
from tqdm import tqdm

from openai import OpenAI
from score.score import score

def process_description_batch(batch_number):
    """
    单个任务：请求 deepseek-chat API 生成一批描述，并返回答案列表（不在此处做去重）。
    """
    client = OpenAI(api_key='sk-8a652d9cc48342789bd2657f9be44c2e', base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": """
                    You're a creative assistant tasked with generating textual descriptions similar to the following examples.
                    Examples:
                    - a starlit night over snow-covered peaks
                    - black and white checkered pants
                    - crimson rectangles forming a chaotic grid
                    - burgundy corduroy pants with patch pockets and silver buttons
                    - orange corduroy overalls
                    - a lighthouse overlooking the ocean
                    - a green lagoon under a cloudy sky
                    - a snowy plain
                    - a maroon dodecahedron interwoven with teal threads
                    - a purple silk scarf with tassel trim
                    - magenta trapezoids layered on a translucent silver sheet
                    - gray wool coat with a faux fur collar
                    - a purple forest at dusk
                    - purple pyramids spiraling around a bronze cone
                    - khaki triangles and azure crescents

                    Please generate {number} new, unique, and similarly styled descriptions. Wrap each description with <answer></answer> tags:
                    """.format(number=batch_number)},
        ],
        stream=False
    )

    response_text = response.choices[0].message.content
    answers = extract_answers(response_text)
    
    # 此处不做去重，仅返回所有答案（去除前后空格）
    new_answers = [answer.strip() for answer in answers if answer.strip()]
    for ans in new_answers:
        print(f"Generated description: {ans}")
    return new_answers

def deepseek_descrption_pipeline(number=1000, batch_number=100, filename='descriptions.csv'):
    """
    多线程版生成描述，分多个 batch 请求 deepseek-chat 接口，然后将全局去重后的结果写入 CSV 文件。
    """
    total_batches = number // batch_number
    all_results = []  # 用于存放所有生成的描述

    # 使用 ThreadPoolExecutor 并行请求
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_description_batch, batch_number) for _ in range(total_batches)]
        for future in tqdm(concurrent.futures.as_completed(futures), total=total_batches, desc="Generating descriptions"):
            batch_results = future.result()
            all_results.extend(batch_results)

    # 全局去重：只保留唯一的描述
    unique_results = []
    seen = set()
    for description in all_results:
        d = description.strip()
        if d and d not in seen:
            seen.add(d)
            unique_results.append(d)

    # 将唯一描述写入 CSV，依次编号
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'description'])
        for idx, description in enumerate(unique_results, start=1):
            writer.writerow([idx, description])

    print(f"✅ 转换完成，生成 {filename} 文件")

def process_svg_item(obj_item):
    """
    单个任务：根据描述生成 SVG。返回 (id, svg) 的元组，若生成失败则返回 None。
    """
    item_id = obj_item["id"]
    description = obj_item["description"]
    client = OpenAI(api_key='sk-8a652d9cc48342789bd2657f9be44c2e', base_url="https://api.deepseek.com")
    messages = generate_svg_prompt(description)
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=False
    )
    response_text = response.choices[0].message.content
    svg_response = extract_svg(response_text)
    if svg_response is None:
        return None
    clear_response = enforce_constraints(svg_response)
    if clear_response is None:
        return None
    print("🤖 Clear Response: ", clear_response)
    return (item_id, clear_response)

def deepseek_svg_pipeline(descrptions, filename='svg.csv'):
    """
    多线程版生成 SVG，根据传入的描述对象数组（包含 id 和 description 字段），调用 deepseek-chat 接口生成 SVG 代码，
    最后将成功生成的结果写入 CSV 文件。
    """
    results = []
    # 使用 ThreadPoolExecutor 并行处理每个描述
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 为每个描述提交任务
        futures = {executor.submit(process_svg_item, item): item for item in descrptions}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(descrptions), desc="Generating SVG Code"):
            res = future.result()
            if res is not None:
                results.append(res)

    # 将生成的 SVG 写入 CSV 文件
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'svg'])
        for item_id, svg in results:
            writer.writerow([item_id, svg])

    print(f"✅ 转换完成，生成 {filename} 文件")

def read_annotation_to_csv(filename, output_csv1='output.csv', output_csv2='output2.csv'):
    descpriptions = read_from_json(filename=filename)

    f = open(output_csv1, 'w', newline='', encoding='utf-8')

    f2 = open(output_csv2, 'w', newline='', encoding='utf-8')
    writer = csv.writer(f)
    write2 = csv.writer(f2)
    writer.writerow(['id', 'svg'])
    write2.writerow(['id', 'description'])

    id = 0

    for item in descpriptions:
        svg = item.get('svg')
        try:
            svg_string = read_svg_as_string(svg)
        except Exception as e:
            print(e)
            svg_string = default_svg
        for sentence in item.get('sentences', []):
            writer.writerow([id, svg_string])
        
            write2.writerow([id, sentence])
            id += 1

    print(f"✅ 转换完成，生成 {output_csv1} {output_csv2}文件")



def process_csv():
    svg = read_from_csv(filename="svg.csv")
    description = read_from_csv(filename="description.csv")

    svg['svg'] = svg['svg'].astype(str)
    description['description'] = description['description'].astype(str)
    
    total_svg_res, total_res, mean_res = score(description, svg, 'id')

    print(total_svg_res, total_res, mean_res)

def main_pipeline(
    svg_csv='svg.csv',
    description_csv='description.csv',
    score_csv='score.csv',
    number=1000,
    batch_number=10,
):
    # 检查 description_csv 文件是否存在，若不存在则执行生成
    if not os.path.exists(description_csv):
        deepseek_descrption_pipeline(number=number, batch_number=batch_number, filename=description_csv)
    else:
        print(f"{description_csv} 已存在，跳过生成。")

    # 读取 description_csv 文件
    description_data = read_from_csv(filename=description_csv)
    descriptions = description_data.to_dict(orient='records')

    # 检查 svg_csv 文件是否存在，若不存在则执行生成
    if not os.path.exists(svg_csv):
        deepseek_svg_pipeline(descrptions=descriptions, filename=svg_csv)
    else:
        print(f"{svg_csv} 已存在，跳过生成。")

    if not os.path.exists(score_csv):
        svg_csv = read_from_csv(filename=svg_csv)

        description_csv = read_from_csv(filename=description_csv)

        svg_csv['svg'] = svg_csv['svg'].astype(str)
        description_csv['description'] = description_csv['description'].astype(str)
        
        score(description_csv, svg_csv, 'id', True, score_csv)
        pass
    else:
        print(f"{score_csv} 已存在，跳过生成。")

    

    
