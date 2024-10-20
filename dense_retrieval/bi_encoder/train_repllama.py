

import argparse
from datetime import datetime
import logging
from typing import Iterable

from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    LoggingHandler,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.models import Transformer
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LlamaModel, BitsAndBytesConfig

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
#### /print debug information to stdout



class RepLLaMA(Transformer):

    def __init__(self, **kwargs):
        super(RepLLaMA, self).__init__(**kwargs)
        
        self.tokenizer = self._load_tokenizer(
            kwargs.get("model_name_or_path"),
            kwargs.get("cache_dir")
        )

    def _load_tokenizer(self, model_name_or_path, cache_dir):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer


    def _load_model(self, model_name_or_path, config, cache_dir, backend, **model_args) -> None:
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
        self.auto_model = LlamaModel.from_pretrained(
            model_name_or_path, 
            config=config, 
            cache_dir=cache_dir, 
            quantization_config=bnb_config,
            **model_args
        )

    def _encode(self, input_):
        if input_ is None:
            return None
        
        output = self.auto_model(**input_, output_hidden_states=True)
        hidden = output.hidden_states[-1]
        attention_mask = input_["attention_mask"]

        # paddingではない最後のトークンに対応する埋め込み表現を取得
        ## 行ごとにpaddingされていない部分=実際のトークンの数を集計
        sequence_lengths = attention_mask.sum(dim=1)
        last_token_indices = sequence_lengths - 1

        ## バッチ内の各系列の隠れ層の出力
        ### hidden: (batch_size, seq_len, hidden_size)
        reps = hidden[torch.arange(hidden.size(0)), last_token_indices.squeeze(-1)]
        reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    def forward(self, features: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        trans_features = {
            "input_ids": features["input_ids"],
            "attention_mask": features["attention_mask"]
        }
        trans_features.update({
            "token_embeddings": self._encode(features),
        })

        return trans_features


class InfoNCELoss(torch.nn.Module):
    def __init__(self, model, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.model = model
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        reps = [self.model(sentence_feature)["token_embeddings"] for sentence_feature in sentence_features]
        q_reps = reps[0]
        p_reps = torch.cat(reps[1:])

        # クエリとドキュメントのコサイン類似度行列を計算
        similarity = torch.matmul(q_reps, p_reps.transpose(0, 1))
        similarity = similarity.view(q_reps.size(0), -1)
        
        # 対角成分が正例を表し、それ以外が負例となる
        target = torch.arange(similarity.size(0), device=similarity.device, dtype=torch.long)
        target = target * (p_reps.size(0) // q_reps.size(0))
        
        # クロスエントロピーロスで正例の類似度を最大化
        loss = self.cross_entropy(similarity, target)
        return loss

def add_suffix(example):
    example['query'] = f'query: {example['query']}</s>'
    example['positive'] = f'passage: {example['positive']}</s>'
    example['negative'] = f'passage: {example['negative']}</s>'
    return example

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--max_seq_length", default=300, type=int)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--max_passages", default=0, type=int)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--pooling", default="mean")
    parser.add_argument(
        "--negs_to_use",
        default=None,
        help="From which systems should negatives be used? Multiple systems separated by comma. None = all",
    )
    parser.add_argument("--warmup_steps", default=1000, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--num_negs_per_system", default=5, type=int)
    parser.add_argument("--use_pre_trained_model", default=False, action="store_true")
    parser.add_argument("--use_all_queries", default=False, action="store_true")
    parser.add_argument("--ce_score_margin", default=3.0, type=float)
    args = parser.parse_args()

    print(args)

    # The  model we want to fine-tune
    model_name = args.model_name

    train_batch_size = (
        args.train_batch_size
    )  # Increasing the train batch size improves the model performance, but requires more GPU memory
    max_seq_length = args.max_seq_length  # Max length for passages. Increasing it, requires more GPU memory
    ce_score_margin = args.ce_score_margin  # Margin for the CrossEncoder score between negative and positive passages
    num_negs_per_system = (
        args.num_negs_per_system
    )  # We used different systems to mine hard negatives. Number of hard negatives to add from each system
    num_epochs = args.epochs  # Number of epochs we want to train

    # Load our embedding model
    logging.info("Create LLaMa model")
    word_embedding_model = RepLLaMA(model_name_or_path=model_name, max_seq_length=max_seq_length)
    model = SentenceTransformer(modules=[word_embedding_model])
        
    output_dir = "output/train_repllama-{}-margin_{:.1f}-{}".format(
        model_name.replace("/", "-"), ce_score_margin, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )

    train_dataset = load_dataset("sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1", "triplet-hard", split="train").select(range(10000))
    train_dataset = train_dataset.map(add_suffix)
    train_loss = InfoNCELoss(model=model)

    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=output_dir,
        # Optional training parameters:
        num_train_epochs=1,
        per_device_train_batch_size=train_batch_size,
        # per_device_eval_batch_size=train_batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        # Optional tracking/debugging parameters:
        # eval_strategy="steps",
        # eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=100,
        run_name="sentence-transformer-repllama",  # Will be used in W&B if `wandb` is installed
        # report_to="wandb"
    )
    # dev_evaluator = TripletEvaluator(
    #     anchors=eval_dataset["anchor"],
    #     positives=eval_dataset["positive"],
    #     negatives=eval_dataset["negative"],
    #     name="msmarco-bm25",
    # )
    # dev_evaluator(model)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        loss=train_loss,
        # evaluator=dev_evaluator,
    )
    trainer.train()

    # Save the model
    model.save(output_dir)