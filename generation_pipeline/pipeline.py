from generation_pipeline.prompt import generate_descrption_prompt, generate_svg_prompt
from generation_pipeline.util import enforce_constraints, extract_answers, prompt_with_deepseek, extract_svg, save_to_json, save_to_svg
from tqdm import tqdm

from openai import OpenAI


def deepseek_descrption_pipeline(number=1000, batch_number=100, filename='descriptions.txt'):
    batch = number // batch_number

    for _ in tqdm(range(batch), desc="Generating descriptions"):
        # description = generate_descrption_prompt(number=batch_number)
        # import pdb; pdb.set_trace()


        client = OpenAI(api_key='sk-8a652d9cc48342789bd2657f9be44c2e', base_url="https://api.deepseek.com")

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                { 
                    "role": "user", 
                    "content": """
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

                    Please generate 20 new, unique, and similarly styled descriptions. Wrap each description with <answer></answer> tags:
                    """,
                } 
            ],
            stream=False
        )

        response = response.choices[0].message.content

        # response = prompt_with_deepseek(description)
        answers = extract_answers(response)

        with open(filename, 'a+', encoding='utf-8') as file:
            for answer in answers:
                file.write(answer.strip() + '\n')



def deepseek_svg_pipeline(descrptions):

    for obj_item in tqdm(descrptions, desc="Generating SVG Code"):
        # description = generate_descrption_prompt(number=batch_number)
        # import pdb; pdb.set_trace()

        id = obj_item["label"]

        description = obj_item["sentences"]


        client = OpenAI(api_key='sk-8a652d9cc48342789bd2657f9be44c2e', base_url="https://api.deepseek.com")

        messages = generate_svg_prompt(description)

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=False
        )

        response = response.choices[0].message.content

        response = extract_svg(
            response
        )

        if response is None:
            continue

        print("🤖 Response: ", response)

        clear_response = enforce_constraints(response)

        print("🤖 Clear Response: ", clear_response)

        if clear_response is None:
            continue

        save_to_svg(clear_response, filename=f"svg/{id}.svg")
    

