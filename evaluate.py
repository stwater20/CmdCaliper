import argparse
import collections
import os

from datasets import load_dataset
import torch
import torch.nn.functional as F

from src.inferencer import SentenceTransformerInferencer

class RetrievalEvaluator:
    DATASET_NAME = "CyCraftAI/CyPHER"
    def __init__(self, batch_size, device):
        raw_test_data = load_dataset(self.DATASET_NAME)
        self.test_data = []
        for d in raw_test_data["test"]:
            self.test_data.append(d)
        self.batch_size = batch_size
        self.device = device

    def evaluate(self, inferencer):
        with torch.no_grad():
            evaluate_metric = self._evaluate(inferencer)
        return evaluate_metric

    def _batch_inference(self, sentence_list, inferencer):
        embedding_list = []
        for i in range(0, len(sentence_list), self.batch_size):
            sub_sentence_list = sentence_list[i: min(i+self.batch_size, len(sentence_list))]
            sub_embedding_list = inferencer(sub_sentence_list).to(self.device)
            
            embedding_list.append(sub_embedding_list)
        embedding_list = torch.cat(embedding_list, 0)
        return embedding_list

    def _evaluate(self, inferencer):
        testing_query_cmd_list = [d["query_cmd"] for d in self.test_data]
        testing_positive_cmd_list = [d["positive_cmd"] for d in self.test_data]

        testing_query_cmd_embedding_list = self._batch_inference(testing_query_cmd_list, inferencer)
        testing_positive_cmd_embedding_list = self._batch_inference(testing_positive_cmd_list, inferencer)

        evaluate_positive_rank_list = []
        for i, d in enumerate(self.test_data):
            negative_key_cmd_list = [testing_query_cmd_list[j] for j in d["negative_index_list"]]
            negative_key_embedding_list = torch.stack([testing_query_cmd_embedding_list[j] for j in d["negative_index_list"]], 0)

            query_cmd_embedding = testing_query_cmd_embedding_list[i].unsqueeze(0)
            positive_cmd_embedding = testing_positive_cmd_embedding_list[i].unsqueeze(0)

            candidate_cmd_embedding_list = torch.cat([positive_cmd_embedding, negative_key_embedding_list], 0)
            cosine_similarity_list = F.cosine_similarity(query_cmd_embedding, candidate_cmd_embedding_list, -1)
            cosine_similarity_sorted_index_list = cosine_similarity_list.argsort(descending=True)
            positive_rank = torch.where(cosine_similarity_sorted_index_list == 0)[0].tolist()[0]

            evaluate_positive_rank_list.append(positive_rank)
        mrr_metrics, topk_metrics = self._calculate_metrics(evaluate_positive_rank_list)
        return mrr_metrics, topk_metrics

    def _calculate_metrics(self, rank_list):
        mrr_metrics = collections.defaultdict(int)
        topk_metrics = collections.defaultdict(int)

        for rank in rank_list:
            for k in [3, 10]:
                if rank + 1 > k:
                    continue
                mrr_metrics[k] += 1.0 / (rank + 1) / len(rank_list)
                topk_metrics[k] += 1 / len(rank_list)
        return dict(mrr_metrics), dict(topk_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    evaluator = RetrievalEvaluator(args.batch_size, args.device)
    print(f"Using the SentenceTransformerInferencer for the model - {args.model_name}")
    inferencer = SentenceTransformerInferencer(
        args.model_name, args.device
    )

    print(f"The performance of the model - {args.model_name}: {evaluator.evaluate(inferencer)}")

