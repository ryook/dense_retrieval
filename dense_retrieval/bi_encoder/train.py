"""
This examples show how to train a Bi-Encoder for the MS Marco dataset (https://github.com/microsoft/MSMARCO-Passage-Ranking).

The queries and passages are passed independently to the transformer network to produce fixed sized embeddings.
These embeddings can then be compared using cosine-similarity to find matching passages for a given query.

For training, we use MultipleNegativesRankingLoss. There, we pass triplets in the format:
(query, positive_passage, negative_passage)

Negative passage are hard negative examples, that were mined using different dense embedding methods and lexical search methods.
Each positive and negative passage comes with a score from a Cross-Encoder. This allows denoising, i.e. removing false negative
passages that are actually relevant for the query.

With a distilbert-base-uncased model, it should achieve a performance of about 33.79 MRR@10 on the MSMARCO Passages Dev-Corpus

Running this script:
python train_bi-encoder-v3.py
"""

import argparse
import logging
from datetime import datetime

from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler, SentenceTransformer, losses, models

from data import MSMARCODataset
from data_utils import read_corpus, read_queries, generate_train_queries



#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
#### /print debug information to stdout



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_batch_size", default=64, type=int)
    parser.add_argument("--max_seq_length", default=300, type=int)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--max_passages", default=0, type=int)
    parser.add_argument("--epochs", default=10, type=int)
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
    if args.use_pre_trained_model:
        logging.info("use pretrained SBERT model")
        model = SentenceTransformer(model_name)
        model.max_seq_length = max_seq_length
    else:
        logging.info("Create new SBERT model")
        word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), args.pooling)
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    model_save_path = "output/train_bi-encoder-mnrl-{}-margin_{:.1f}-{}".format(
        model_name.replace("/", "-"), ce_score_margin, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )


    corpus = read_corpus()
    queries = read_queries()
    train_queries = generate_train_queries(queries, args, ce_score_margin)
    # For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
    train_dataset = MSMARCODataset(train_queries, corpus=corpus)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)

    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=args.warmup_steps,
        use_amp=True,
        checkpoint_path=model_save_path,
        checkpoint_save_steps=len(train_dataloader),
        optimizer_params={"lr": args.lr},
    )

    # Save the model
    model.save(model_save_path)