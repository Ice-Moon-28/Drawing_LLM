
from openai import OpenAI
from setting import Deepseek_Api_Key, OpenAI_Api_Key


def prompt_with_deepseek(description):
    client = OpenAI(api_key=Deepseek_Api_Key, base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        # messages=[
        #     {"role": "system", "content": "You are a helpful assistant"},
        #     {"role": "user", "content": "Hello"},
        # ],
        messages=description,
        stream=False
    )

    return response.choices[0].message.content


def prompt_with_openai1(description):
    client = OpenAI(api_key=OpenAI_Api_Key)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        # messages=[
        #     {"role": "system", "content": "You are a helpful assistant."},
        #     {"role": "user", "content": description},
        # ],
        messages=description,
        stream=False
    )

    return response.choices[0].message.content