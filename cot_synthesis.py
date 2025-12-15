import asyncio
import aiohttp
import os
import json
from tqdm.asyncio import tqdm
from datetime import datetime
from datasets import load_dataset
import argparse


parser = argparse.ArgumentParser(description="Script for vLLM inference.")
parser.add_argument("-d", "--dir", type=str, default="")
parser.add_argument("-i", "--index", type=int, default=0)
parser.add_argument("-m", "--model_name", type=int, default=0)
args = parser.parse_args()

TIMESTAMP = datetime.now().strftime(f"%Y%m%d%H%M%S")
DIR_PATH = f"./outputs/responses_{args.model_name}_{TIMESTAMP}"
if not os.path.exists(DIR_PATH):
    os.makedirs(DIR_PATH)


def get_messages(datum):
    image_base64    = datum['image']
    conversations   = datum['conversations']
    question        = conversations[0]['value'][7: ]

    messages = [{
        "role":"user",
        "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
        ]
    }]
   
    return messages


async def robust_fetch(session, datum, semaphore):
    async with semaphore:
        messages = get_messages(datum)
        model_name = f"{args.model_name}_{args.index}"
        port = 8000 + 100 * args.index
        payload = {'model': model_name, 'messages': messages}
        
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
                    
                    return result['choices'][0]['message']['content']
                
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
                if attempt < max_retries:
                    await asyncio.sleep(1)
                    continue
                else:
                    return None
                    
            except Exception as e:
                save_error_log(datum, str(e))
                return None
                
        return None


def save_error_log(datum, error_message):
    with open(f'{DIR_PATH}/error_log.jsonl', 'a') as f:
        log_entry = {
            'id': datum['id'],
            'error': error_message[: 300],
        }
        f.write(json.dumps(log_entry) + '\n')
       

async def main():
    DATASET_PATH = args.dir
    DATASET_SPLIT = "train"
    NUM_SHARDS = 18
    dataset = load_dataset(DATASET_PATH, split=DATASET_SPLIT)
    
    for i in range(NUM_SHARDS):
        semaphore = asyncio.Semaphore(128)
        async with aiohttp.ClientSession() as session:
            tasks = [robust_fetch(session, datum, semaphore) for datum in dataset.shard(num_shards=NUM_SHARDS, index=i)]
            results = await tqdm.gather(*tasks, desc='Inferencing...')

            with open(f"{DIR_PATH}/responses_{args.model_name}_{TIMESTAMP}_{i}.json", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    asyncio.run(main())
