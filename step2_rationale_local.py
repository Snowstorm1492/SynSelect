import os
import json
import asyncio
import aiohttp
import argparse
from tqdm.asyncio import tqdm

SYS_PROMPT = """Suppose you are an AI expert proficient in the refinement of the reasoning process.
Your task is to extract the most essential reasoning basis from the lengthy chain of thought (CoT) provided as input, eliminating all redundant information.
The input is the original CoT, and the output is the refined core Rationale. The Rationale must be able to explain how the final answer was derived. 

Requirements:
1. Only output the refined Rationale and remove all redungations.
2. Strictly based on the input text, no additions or assumptions are made.
3. The Rationale should maintain logical integrity and serve as the direct basis for the final answer.
4. The output format is plain text paragraphs."""

USER_PROMPT_TEMPLATE = """Now please process the following input:

Input: {}

Rationale:"""


parser = argparse.ArgumentParser(description="Script for Rationale Extraction.")
parser.add_argument("-d", "--dir", type=str, default="")
parser.add_argument("-i", "--index", type=int, default=0)
parser.add_argument("-m", "--model_name", type=int, default=0)
args = parser.parse_args()

def get_messages(reply):
    messages = [{
        'role': 'system',
        'content': SYS_PROMPT
    }]

    messages.append({
        'role': 'user',
        'content': USER_PROMPT_TEMPLATE.format(reply)
    })
   
    return messages


async def robust_fetch(session, judgement, reply, semaphore):
    if judgement["correctness"] != 1:
        return None
    
    async with semaphore:
        messages = get_messages(reply)
        model_name = f'{args.model_name}_{args.index}'
        port = 8000 + 100 * args.index
        payload = {
            'model': model_name,
            'messages': messages,
            'chat_template_kwargs': {"enable_thinking": False}
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
                    
                    return result['choices'][0]['message']['content']
                
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
                if attempt < max_retries:
                    await asyncio.sleep(1)
                    continue
                else:
                    return None
                    
            except Exception as e:
                print(str(e))
                return None
                
        return None
       

async def main(dir_path):
    NUM_SHARDS = 18
    replies = list()

    with open(f"{dir_path}/evaluation1-{os.path.basename(dir_path)}.json", "r", encoding="utf-8") as f:
        judgements = json.load(f)
    
    for i in range(NUM_SHARDS):
        with open(f"{dir_path}/{os.path.basename(dir_path)}_{i}.json", "r", encoding="utf-8") as f:
            replies.extend(json.load(f))
    
    semaphore = asyncio.Semaphore(128)
    async with aiohttp.ClientSession() as session:
        tasks = [robust_fetch(session, judgement, reply, semaphore) for judgement, reply in zip(judgements, replies)]
        results = await tqdm.gather(*tasks, desc='Extraction')
        
    for judgement, result in zip(judgements, results):
        judgement["rationale"] = result

    with open(f"{dir_path}/evaluation2-{os.path.basename(dir_path)}.json", "w", encoding="utf-8") as f:
        json.dump(judgements, f, ensure_ascii=False, indent=2)


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
