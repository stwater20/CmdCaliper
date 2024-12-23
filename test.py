import argparse
import torch
import torch.nn.functional as F
from datasets import load_dataset
from src.inferencer import SentenceTransformerInferencer
import random

class CommandRetriever:
    DATASET_NAME = "CyCraftAI/CyPHER"

    def __init__(self, batch_size, device):
        raw_data = load_dataset(self.DATASET_NAME)
        self.data = raw_data["test"]
        self.batch_size = batch_size
        self.device = device

    def retrieve_similar_commands(self, query_cmd, inferencer, top_k=5):
        with torch.no_grad():
            query_embedding = inferencer([query_cmd]).to(self.device)
            dataset_cmds = [d["query_cmd"] for d in self.data]
            dataset_embeddings = self._batch_inference(dataset_cmds, inferencer)

            cosine_similarities = F.cosine_similarity(query_embedding, dataset_embeddings, dim=-1)
            top_k_indices = cosine_similarities.argsort(descending=True)[:top_k]

            similar_cmds = [(dataset_cmds[i], cosine_similarities[i].item()) for i in top_k_indices]
            return similar_cmds

    def _batch_inference(self, sentence_list, inferencer):
        embeddings = []
        for i in range(0, len(sentence_list), self.batch_size):
            sub_sentences = sentence_list[i: min(i + self.batch_size, len(sentence_list))]
            sub_embeddings = inferencer(sub_sentences).to(self.device)
            embeddings.append(sub_embeddings)
        return torch.cat(embeddings, dim=0)

    def generate_cmd_from_description(self, description, inferencer, generation_num=3):
        cmd_examples = [d["query_cmd"] for d in self.data[:5]]  # Take 5 examples for context
        prompt = self._get_cmd_generation_prompt(cmd_examples, description, generation_num)
        # Simulate GPT-based generation (replace this with actual GPT API call if needed)
        generated_cmds = self._simulate_gpt_generation(prompt, generation_num)
        return generated_cmds

    def _get_cmd_generation_prompt(self, cmd_examples, description, generation_num):
        cmd_prompt = "\n".join([f"- {c}" for c in cmd_examples])
        prompt = f"""Your task is to generate {generation_num} Windows command lines based on the following description and example commands.

Description:
{description}

Command Line Examples:
{cmd_prompt}

Synthesize new command lines that align with the described purpose but appear diverse in argument value, format, and length. Be creative, and prioritize realistic and executable commands.
Please provide only the generated commands prefixed with "<CMD>" and separate each command with a newline.
"""
        return prompt

    def _simulate_gpt_generation(self, prompt, generation_num):
        # This function simulates GPT's output (for illustration purposes)
        simulated_cmds = [
            f"<CMD> icacls C:\\Test\\Directory /grant User:F",
            f"<CMD> robocopy C:\\Source D:\\Destination /E",
            f"<CMD> net user NewUser Password123 /add"
        ]
        return simulated_cmds[:generation_num]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--query-cmd", type=str, help="The command line query to search for similar commands.")
    parser.add_argument("--description", type=str, help="The description to generate commands from.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of similar commands to return.")
    parser.add_argument("--generation-num", type=int, default=3, help="Number of commands to generate from description.")
    args = parser.parse_args()

    retriever = CommandRetriever(args.batch_size, args.device)
    print(f"Loading model: {args.model_name}")
    inferencer = SentenceTransformerInferencer(
        args.model_name, args.device
    )

    if args.query_cmd:
        print(f"Querying for similar commands to: {args.query_cmd}")
        similar_cmds = retriever.retrieve_similar_commands(args.query_cmd, inferencer, args.top_k)
        print("Top similar commands:")
        for cmd, score in similar_cmds:
            print(f"Command: {cmd}, Similarity Score: {score:.4f}")

    if args.description:
        print(f"Generating commands from description: {args.description}")
        generated_cmds = retriever.generate_cmd_from_description(args.description, inferencer, args.generation_num)
        print("Generated commands:")
        for cmd in generated_cmds:
            print(cmd)
