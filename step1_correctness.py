import asyncio
import aiohttp
import os
import json
import argparse
from tqdm.asyncio import tqdm
from datasets import load_dataset

SYS_PROMPT = "Suppose you are an AI judge. Below, the user will provide two pieces of text. Among them, the first paragraph is a fragment of the LLM's response process to a question; The second paragraph of text is the answer result of this question. Now, could you please, based on these two paragraphs of text, determine whether the large model's response is correct? Your answer should be given strictly in the following form: If correct, reply:\n\n\"\"\"\nrrrrr rrrrr rrrrr!!!\n\nBingo!!!\n\"\"\"\n\nIf the answer is wrong, reply:\n\n\"\"\"\nxxxxx xxxxx xxxxx!!!\n\nWrong !!!\n\"\"\"\n\nIf it cannot be judged based on the available information, reply:\n\n\"\"\"\nooooo ooooo ooooo!!!\n\nI do not know!!!\n\"\"\"\n"
USER_PROMPT_TEMPLATE = "The first paragraph of the text, that is, the response of LLM, is:\n\n------------------------------\n{}\n------------------------------\n\n\nThe second paragraph of the text, that is, the ground-truth answer result, is:\n\n------------------------------\n{}\n------------------------------\n"

parser = argparse.ArgumentParser(description="Script for vLLM inference.")
parser.add_argument("-d", "--dir", type=str, default="")
parser.add_argument("-i", "--index", type=int, default=0)
parser.add_argument("-m", "--model_name", type=int, default=0)
args = parser.parse_args()

def get_messages(datum, reply):
    text1 = reply[-200: ]
    text2 = datum['conversations'][1]['value'][-100: ]

    messages = [{
        'role': 'system',
        'content': SYS_PROMPT
    }]

    messages.append({
        'role': 'user',
        'content': USER_PROMPT_TEMPLATE.format(text1, text2)
    })
   
    return messages


async def fetch(session, datum, reply, semaphore):
    if reply is None:
        return ''
    
    async with semaphore:
        messages = get_messages(datum, reply)
        async with session.post(f'http://localhost:{8000 + 100 * args.index}/v1/chat/completions', json={
            'model': f'{args.model_name}_{args.index}',
            'messages': messages,
            'temperature': 0.2,
            'max_tokens': 2048
        }) as response:
            result = await response.json()
            if 'choices' not in result.keys():
                print(f"!!! Exception occured: {datum['index']}. Resuming...")
                return ''
            else:
                return result['choices'][0]['message']['content']
       

async def main(dir_path):
    DATASET_PATH = args.dir
    DATASET_SPLIT = "train"
    NUM_SHARDS = 18
    dataset = load_dataset(DATASET_PATH, split=DATASET_SPLIT)
    output_list = list()
    
    for i in range(NUM_SHARDS):
        with open(f"{dir_path}/{os.path.basename(dir_path)}_{i}.json", "r", encoding="utf-8") as f:
            replies = json.load(f)
            dataset_shard = dataset.shard(num_shards=NUM_SHARDS, index=i)
            
            semaphore = asyncio.Semaphore(2048)
            async with aiohttp.ClientSession() as session:
                tasks = [fetch(session, datum, reply, semaphore) for datum, reply in zip(dataset_shard, replies)]
                results = await tqdm.gather(*tasks, desc='Inferencing...')

        for datum, result in zip(dataset_shard, results):
            if "rrrrr" in result or "Bingo" in result:
                correctness = 1
            elif "xxxxx" in result or "Wrong" in result:
                correctness = 2
            elif "ooooo" in result or "I do not know" in result:
                correctness = 3
            else:
                correctness = 0

            output_list.append({
                "id": datum['id'],
                "correctness": correctness,
                "rationale": None,
                "ppl": None,
            })
                
    with open(f"{dir_path}/evaluation1-{os.path.basename(dir_path)}.json", "w", encoding="utf-8") as f:
        json.dump(output_list, f, ensure_ascii=False, indent=2)


if __name__ == "__main__": 

    DIR_LIST = [
        [
            "./SynAgent0/outputs/responses_SynAgent0_0",
            "./SynAgent0/outputs/responses_SynAgent0_1",
            "./SynAgent0/outputs/responses_SynAgent0_2",
            "./SynAgent0/outputs/responses_SynAgent0_3",
            "./SynAgent0/outputs/responses_SynAgent0_4",
            "./SynAgent0/outputs/responses_SynAgent0_5",
        ],
        [
            "./SynAgent1/outputs/responses_SynAgent1_0",
            "./SynAgent1/outputs/responses_SynAgent1_1",
            "./SynAgent1/outputs/responses_SynAgent1_2",
            "./SynAgent1/outputs/responses_SynAgent1_3",
            "./SynAgent1/outputs/responses_SynAgent1_4",
            "./SynAgent1/outputs/responses_SynAgent1_5",
        ],
        [
            "./SynAgent2/outputs/responses_SynAgent2_0",
            "./SynAgent2/outputs/responses_SynAgent2_1",
            "./SynAgent2/outputs/responses_SynAgent2_2",
            "./SynAgent2/outputs/responses_SynAgent2_3",
            "./SynAgent2/outputs/responses_SynAgent2_4",
            "./SynAgent2/outputs/responses_SynAgent2_5",
        ]
    ]

    asyncio.run(main(DIR_LIST[0][args.index]))
    asyncio.run(main(DIR_LIST[1][args.index]))
    asyncio.run(main(DIR_LIST[2][args.index]))
