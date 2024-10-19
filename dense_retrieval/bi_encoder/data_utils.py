import os
import logging
import gzip
import json
import pickle
import tarfile

import tqdm
from sentence_transformers import LoggingHandler, util


logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)

### Now we read the MS Marco dataset
data_folder = "msmarco-data"

def read_corpus():
    #### Read the corpus files, that contain all the passages. Store them in the corpus dict
    corpus = {}  # dict in the format: passage_id -> passage. Stores all existent passages
    collection_filepath = os.path.join(data_folder, "collection.tsv")
    if not os.path.exists(collection_filepath):
        tar_filepath = os.path.join(data_folder, "collection.tar.gz")
        if not os.path.exists(tar_filepath):
            logging.info("Download collection.tar.gz")
            util.http_get("https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz", tar_filepath)

        with tarfile.open(tar_filepath, "r:gz") as tar:
            tar.extractall(path=data_folder)

    logging.info("Read corpus: collection.tsv")
    with open(collection_filepath, encoding="utf8") as fIn:
        for line in fIn:
            pid, passage = line.strip().split("\t")
            pid = int(pid)
            corpus[pid] = passage
    return corpus


def read_queries() -> dict:
    ### Read the train queries, store in queries dict
    queries = {}  # dict in the format: query_id -> query. Stores all training queries
    queries_filepath = os.path.join(data_folder, "queries.train.tsv")
    if not os.path.exists(queries_filepath):
        tar_filepath = os.path.join(data_folder, "queries.tar.gz")
        if not os.path.exists(tar_filepath):
            logging.info("Download queries.tar.gz")
            util.http_get("https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz", tar_filepath)

        with tarfile.open(tar_filepath, "r:gz") as tar:
            tar.extractall(path=data_folder)


    with open(queries_filepath, encoding="utf8") as fIn:
        for line in fIn:
            qid, query = line.strip().split("\t")
            qid = int(qid)
            queries[qid] = query

    return queries



def generate_train_queries(queries: dict, args, ce_score_margin: float = 3.0) -> dict:
    # Load a dict (qid, pid) -> ce_score that maps query-ids (qid) and paragraph-ids (pid)
    # to the CrossEncoder score computed by the cross-encoder/ms-marco-MiniLM-L-6-v2 model
    ce_scores_file = os.path.join(data_folder, "cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz")
    if not os.path.exists(ce_scores_file):
        logging.info("Download cross-encoder scores file")
        util.http_get(
            "https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz",
            ce_scores_file,
        )

    logging.info("Load CrossEncoder scores dict")
    with gzip.open(ce_scores_file, "rb") as fIn:
        ce_scores = pickle.load(fIn)

    # As training data we use hard-negatives that have been mined using various systems
    hard_negatives_filepath = os.path.join(data_folder, "msmarco-hard-negatives.jsonl.gz")
    if not os.path.exists(hard_negatives_filepath):
        logging.info("Download cross-encoder scores file")
        util.http_get(
            "https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/msmarco-hard-negatives.jsonl.gz",
            hard_negatives_filepath,
        )


    logging.info("Read hard negatives train file")
    train_queries = {}
    negs_to_use = None
    with gzip.open(hard_negatives_filepath, "rt") as fIn:
        for line in tqdm.tqdm(fIn):
            data = json.loads(line)

            # Get the positive passage ids
            qid = data["qid"]
            pos_pids = data["pos"]

            if len(pos_pids) == 0:  # Skip entries without positives passages
                continue

            pos_min_ce_score = min([ce_scores[qid][pid] for pid in data["pos"]])
            ce_score_threshold = pos_min_ce_score - ce_score_margin

            # Get the hard negatives
            neg_pids = set()
            if negs_to_use is None:
                if args.negs_to_use is not None:  # Use specific system for negatives
                    negs_to_use = args.negs_to_use.split(",")
                else:  # Use all systems
                    negs_to_use = list(data["neg"].keys())
                logging.info("Using negatives from the following systems: {}".format(", ".join(negs_to_use)))

            for system_name in negs_to_use:
                if system_name not in data["neg"]:
                    continue

                system_negs = data["neg"][system_name]
                negs_added = 0
                for pid in system_negs:
                    if ce_scores[qid][pid] > ce_score_threshold:
                        continue

                    if pid not in neg_pids:
                        neg_pids.add(pid)
                        negs_added += 1
                        if negs_added >= args.num_negs_per_system:
                            break

            if args.use_all_queries or (len(pos_pids) > 0 and len(neg_pids) > 0):
                train_queries[data["qid"]] = {
                    "qid": data["qid"],
                    "query": queries[data["qid"]],
                    "pos": pos_pids,
                    "neg": neg_pids,
                }

    del ce_scores

    logging.info(f"Train queries: {len(train_queries)}")

    return train_queries