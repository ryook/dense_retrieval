# %%
import sys
import os

sys.path.append(os.path.abspath(os.path.join('..')))

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn, Tensor
from transformers import PreTrainedModel, AutoModel, LlamaModel, AutoTokenizer
from transformers.file_utils import ModelOutput
from peft import LoraConfig, get_peft_model, TaskType


from data import HFTrainDataset, TrainDataset, TrainCollator
from tevatron.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments
from trainer import TevatronTrainer

# %%
@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class EncoderModel(nn.Module):
    def __init__(self, lm_q: PreTrainedModel, lm_p: PreTrainedModel, pooler: nn.Module=None):
        super(EncoderModel, self).__init__()

        self.lm_q = lm_q
        self.lm_p = lm_p
        self.pooler = pooler
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")


    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None):
        q_reps = self.encode_query(query)
        p_reps = self.encode_passage(passage)
        loss = None
        scores = None

        # traingin
        if self.training:
            scores = self.compute_similarity(q_reps, p_reps)

            # 類似度スコアのテンソルをクエリごとの類似度スコアの行列に整形し直す
            # (クエリの数, パッセージの数 / クエリの数)
            scores = scores.view(q_reps.size(0), -1)

            # クエリごとの正解ラベルを整形し直す
            # クエリに対して関連するパッセージの分だけインデックスを調整
            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * (p_reps.size(0) // q_reps.size(0))

            loss = self.compute_loss(scores, target)

        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    def encode_passage(self, passage):
        raise NotImplementedError

    def encode_query(self, query):
        raise NotImplementedError

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)


# %%
class RepLLaMA(EncoderModel):
    def __init__(self, lm_q: PreTrainedModel, lm_p: PreTrainedModel, pooler: nn.Module=None):
        super(RepLLaMA, self).__init__(lm_q, lm_p, pooler)

    def encode_passage(self, passage):
        if passage is None:
            return None

        passage_output = self.lm_p(**passage, output_hidden_states=True)
        p_hidden = passage_output.hidden_states[-1]
        attention_mask = passage["attention_mask"]

        # paddingではない最後のトークンに対応する埋め込み表現を取得
        ## 行ごとにpaddingされていない部分=実際のトークンの数を集計
        sequence_lengths = attention_mask.sum(dim=1)
        last_token_indices = sequence_lengths - 1

        ## バッチ内の各系列の隠れ層の出力
        ### p_hidden: (batch_size, seq_len, hidden_size)
        p_reps = p_hidden[torch.arange(p_hidden.size(0)), last_token_indices]
        p_reps = nn.functional.normalize(p_reps, p=2, dim=-1)
        return p_reps

    def encode_query(self, query):
        if query is None:
            return None

        query_output = self.lm_q(**query, output_hidden_states=True)
        q_hidden = query_output.hidden_states[-1]
        attention_mask = query["attention_mask"]

        # paddingではない最後のトークンに対応する埋め込み表現を取得
        ## 行ごとにpaddingされていない部分=実際のトークンの数を集計
        sequence_lengths = attention_mask.sum(dim=1)
        last_token_indices = sequence_lengths - 1

        ## バッチ内の各系列の隠れ層の出力
        ### q_hidden: (batch_size, seq_len, hidden_size)
        q_reps = q_hidden[torch.arange(q_hidden.size(0)), last_token_indices]
        q_reps = nn.functional.normalize(q_reps, p=2, dim=-1)
        return q_reps

    def gradient_checkpointing_enable(self, **kwargs):
        return self.lm_q.base_model.gradient_checkpointing_enable()

    # いる？？
    @staticmethod
    def build_peft_model(peft_model_name: str):
        config = LoraConfig.from_pretrained(peft_model_name)
        config.inference_mode = False
        base_model = LlamaModel.from_pretrained(config.base_model_name_or_path)
        model = get_peft_model(base_model, config)
        model.print_trainable_parameters()
        return model

    @classmethod
    def build(cls, model_config, train_config, **hf_kwargs):
        base_model = LlamaModel.from_pretrained(model_config.model_name_or_path, **hf_kwargs)

        if train_config.gradient_checkpointing:
            base_model.gradient_checkpointing_enable()

        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0


        peft_config = LoraConfig(
            base_model_name_or_path=model_config.model_name_or_path,
            task_type=TaskType.FEATURE_EXTRACTION,
            r=32,
            lora_alpha=64,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
            inference_mode=False
        )

        hf_model = get_peft_model(base_model, peft_config)
        model = cls(
            lm_q=hf_model,
            lm_p=hf_model,
            pooler=None,  
        )
        return model

    def save(self, output_dir):
        self.lm_q.save_pretrained(output_dir)
        

if __name__ == "__main__":
    
    # %%
    model_args = ModelArguments(
        model_name_or_path="meta-llama/Llama-3.2-1B"
    )

    data_args = DataArguments(
        dataset_name="Tevatron/msmarco-passage",
        train_n_passages=16,
        q_max_len=32,
        p_max_len=128,
        dataset_proc_num=32
    )


    training_args = TrainingArguments(
        output_dir="model_repllama",
        save_steps=20,
        learning_rate=1e-4,
        num_train_epochs=1,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        overwrite_output_dir=True,
        report_to="wandb"
    )

    # %%
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )
    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # %%
    model = RepLLaMA.build(
        model_args,
        training_args,
        cache_dir=model_args.cache_dir,
    )

    # %%
    train_dataset = HFTrainDataset(
        tokenizer=tokenizer, 
        data_args=data_args,
        cache_dir=data_args.data_cache_dir or model_args.cache_dir
    )
    train_dataset = TrainDataset(data_args, train_dataset.process(), tokenizer)

    # %%
    trainer = TevatronTrainer(
        model=model,
        args=training_args,
        data_collator=TrainCollator(
            tokenizer,
            max_p_len=data_args.p_max_len,
            max_q_len=data_args.q_max_len
        ),
        train_dataset=train_dataset
    )

    # %%
    train_dataset.trainer = trainer
    trainer.train()


