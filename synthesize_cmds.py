import argparse
import os
import random
import time

from src.inferencer import initialize_llm_pool
from src.prompt_templates import get_single_cmd_generation_prompt
from src.utils import (
    load_json, save_json, extract_cmds, load_yaml,
    postprocess_data_synthesis_response
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-to-credential-config", required=True, type=str)

    parser.add_argument("--path-to-seed-cmds", required=True, type=str)
    parser.add_argument("--path-to-output-dir", required=True, type=str)

    parser.add_argument("--cmd-generation-num", default=4, type=int)
    parser.add_argument("--fewshot-cmd-num", default=12, type=int)
    parser.add_argument("--max-generation-num", default=10000, type=int)

    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--max-output-tokens", default=1024, type=int)

    parser.add_argument("--generation-log-filename", default="cmd_synthesis_generation_logs.json", type=str)
    parser.add_argument("--synthesized-cmd-filename", default="synthesized_cmds.json", type=str)
    args = parser.parse_args()

    credential_config = load_yaml(args.path_to_credential_config)
    llm_pool = initialize_llm_pool(credential_config["llm_pool_info"])
 
    os.makedirs(args.path_to_output_dir, exist_ok=True)
    path_to_generation_logs = os.path.join(args.path_to_output_dir, args.generation_log_filename)
    path_to_synthesized_cmds = os.path.join(args.path_to_output_dir, args.synthesized_cmd_filename)

    seed_cmds = load_json(args.path_to_seed_cmds)
    generation_logs = []
    if os.path.isfile(path_to_generation_logs):
        generation_logs = load_json(path_to_generation_logs)
    if os.path.isfile(path_to_synthesized_cmds):
        seed_cmds = load_json(path_to_synthesized_cmds)

    print(f"Seed CMDs Num: {len(seed_cmds)}")
        
    batch_idx = 0
    while len(seed_cmds) < args.max_generation_num:
        if batch_idx % 10 == 0:
            print(f"CMD Synthesis Progress: {len(seed_cmds)}/{args.max_generation_num}")
        batch_idx += 1
        sub_seed_cmds = random.sample(seed_cmds, k=args.fewshot_cmd_num)
        prompt = get_single_cmd_generation_prompt(sub_seed_cmds, args.cmd_generation_num)
        llm_engine, model_name = random.sample(llm_pool, k=1)[0]
        response = llm_engine.inference(
            prompt, model_name, 
            temperature=args.temperature, 
            max_output_tokens=args.max_output_tokens
        )
        response = postprocess_data_synthesis_response(response)

        new_generated_cmd_list = extract_cmds(response)
        generation_logs.append(
            [response, model_name, new_generated_cmd_list, sub_seed_cmds]
        )
        seed_cmds.extend(new_generated_cmd_list)
        save_json(seed_cmds, path_to_synthesized_cmds)
        save_json(generation_logs, path_to_generation_logs)
