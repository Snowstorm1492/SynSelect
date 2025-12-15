import asyncio
import aiohttp
import os
import json
import argparse
import numpy as np
from tqdm.asyncio import tqdm
from datasets import load_dataset

SYS_PROMPT = "Suppose you are an helpful AI assistant. Next, we will provide a picture and a question about this picture. Please and answer this question based on the given image."
USER_PROMPT = """Question: {}

Your Answer: """

parser = argparse.ArgumentParser(description="Script for vLLM inference.")
parser.add_argument("-d", "--dir", type=str, default="")
parser.add_argument("-i", "--index", type=int, default=0)
parser.add_argument("-m", "--model_name", type=int, default=0)
args = parser.parse_args()



def get_messages(datum):
    image_base64    = datum['image']
    conversations   = datum['conversations']
    question        = conversations[0]['value'][7: ]

    messages = [
        {
            'role': 'system',
            'content': SYS_PROMPT
        },
        {
            "role":"user",
            "content": [
                {"type": "text", "text": USER_PROMPT.format(question)},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
            ]
        }
    ]

   
    return messages


async def robust_fetch(session, datum, reply, judgement, semaphore):
    if judgement["correctness"] != 1:
        return None
    
    async with semaphore:
        messages = get_messages(datum)
        model_name = f'{args.model_name}_{args.index}'
        port = 8000 + 100 * args.index
        payload = {
            'model': model_name,
            'messages': messages,
            'temperature': 0.02,
            "logprobs": True,
        }
        
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                async with session.post(
                    f'http://localhost:{port}/v1/chat/completions',
                    json=payload,
                    timeout=60
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text() if response.content else "No detailed erroe report."
                        raise Exception(f"HTTP Error - {response.status}: {error_text[:200]}")
                    
                    content_type = response.headers.get('Content-Type', '')
                    if content_type != 'application/json':
                        error_text = await response.text()
                        raise Exception(f"Non JSON response: {content_type} - {error_text[:100]}")
                    
                    result = await response.json()
                    
                    if 'choices' not in result or not result['choices']:
                        raise Exception("The response is missing the `choices` field.")
                    
                    logprobs = np.array([logprob_dict['logprob'] for logprob_dict in result['choices'][0]['logprobs']['content']])
                    ppl = np.exp(- logprobs.mean())
                    
                    return {
                        "answer": result['choices'][0]['message']['content'],
                        "ppl_value": ppl,
                        "with_cot": False,
                    }
                
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
                if attempt < max_retries:
                    await asyncio.sleep(1)
                    continue
                else:
                    return None
                    
            except Exception as e:
                return None
                
        return None


async def main(directory):
    DATASET_PATH = args.dir
    DATASET_SPLIT = "train"
    NUM_SHARDS = 18
    dataset = load_dataset(DATASET_PATH, split=DATASET_SPLIT)

    with open(f"{directory}/evaluation1-{os.path.basename(directory)}.json", "r", encoding="utf-8") as f:
        judgements = json.load(f)
    
    current_index = 0
    for i in range(NUM_SHARDS):
        dataset_shard = dataset.shard(num_shards=NUM_SHARDS, index=i)
        with open(f"{directory}/{os.path.basename(directory)}_{i}.json", "r", encoding="utf-8") as f:
            replies = json.load(f)
            
            semaphore = asyncio.Semaphore(64)
            async with aiohttp.ClientSession() as session:
                tasks = [robust_fetch(session, datum, reply, judgements[current_index + j], semaphore) for j, (datum, reply) in enumerate(zip(dataset_shard, replies))]
                results = await tqdm.gather(*tasks, desc='Calculation')
            
            for k, result in enumerate(results):
                judgements[current_index + k]["ppl"] = result

        current_index += len(replies)

    with open(f"{directory}/evaluation3_v1-{os.path.basename(directory)}.json", "w", encoding="utf-8") as f:
        json.dump(judgements, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    dir_list = [
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
    
    asyncio.run(main(directory=dir_list[0][args.index]))
    asyncio.run(main(directory=dir_list[1][args.index]))
    asyncio.run(main(directory=dir_list[2][args.index]))