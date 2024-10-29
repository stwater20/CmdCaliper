import argparse
import os
import random

from src.inferencer import initialize_llm_pool
from src.prompt_templates import get_positive_cmd_generation_prompt
from src.utils import (
    load_json, save_json, extract_cmds, load_yaml,
    postprocess_data_synthesis_response
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-to-credential-config", required=True, type=str)

    parser.add_argument("--path-to-all-cmds", required=True, type=str)
    parser.add_argument("--path-to-output-dir", required=True, type=str)

    parser.add_argument("--cmd-generation-num", default=4, type=int)

    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--max-output-tokens", default=1024, type=int)

    parser.add_argument("--generation-log-filename", default="positive_cmd_synthesis_generation_logs.json", type=str)
    parser.add_argument("--similar-map-filename", default="synthesized_similar_cmd_map.json", type=str)
    args = parser.parse_args()

    credential_config = load_yaml(args.path_to_credential_config)
    llm_pool = initialize_llm_pool(credential_config["llm_pool_info"])

    os.makedirs(args.path_to_output_dir, exist_ok=True)
    path_to_generation_logs = os.path.join(args.path_to_output_dir, args.generation_log_filename)
    path_to_similar_cmd_map = os.path.join(args.path_to_output_dir, args.similar_map_filename)

    similar_cmds_map = {}
    if os.path.isfile(path_to_similar_cmd_map):
        similar_cmds_map = load_json(path_to_similar_cmd_map)

    all_cmds = load_json(args.path_to_all_cmds)
    print(f"All Num of CMDs for Positive Synthesis: {len(all_cmds)}")
    all_cmds = [c for c in all_cmds if c not in similar_cmds_map]
    print(f"Remaining Num of CMDs for Positive Synthesis: {len(all_cmds)}")

    generation_logs = []
    if os.path.isfile(path_to_generation_logs):
        generation_logs = load_json(path_to_generation_logs)

    print(f"Num of CMDs for Positive Synthesis: {len(all_cmds)}")
    for i in range(0, len(all_cmds), args.cmd_generation_num):
        if (i // args.cmd_generation_num) % 10 == 0:
            print(f"Positive CMD Synthesis Progress: {i}/{len(all_cmds)}")
        sub_cmd_list = all_cmds[i: min(i+args.cmd_generation_num, len(all_cmds))]
        prompt = get_positive_cmd_generation_prompt(sub_cmd_list)

        llm_engine, model_name = random.sample(llm_pool, k=1)[0]
        response = llm_engine.inference(
            prompt, model_name, 
            temperature=args.temperature, 
            max_output_tokens=args.max_output_tokens
        )

        synthesized_similar_cmd_list = extract_cmds(response)
        if len(synthesized_similar_cmd_list) != len(sub_cmd_list):
            # Deprecate this generation due to the no match generation num.
            continue

        generation_logs.append([
            response, model_name, synthesized_similar_cmd_list, sub_cmd_list
        ])
        for cmd, synthesized_similar_cmd in zip(sub_cmd_list, synthesized_similar_cmd_list):
            similar_cmds_map[cmd] = synthesized_similar_cmd
        save_json(similar_cmds_map, path_to_similar_cmd_map)
        save_json(generation_logs, path_to_generation_logs)
