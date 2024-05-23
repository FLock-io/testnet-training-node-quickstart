import os

from node import TrainNode


if __name__ == '__main__':
    if os.environ.get("HF_TOKEN") is None:
        raise Exception("HF_TOKEN not found in environment variables.")
    if os.environ.get("FLOCK_API_KEY") is None:
        raise Exception(
            "FLOCK_API_KEY not found in environment variables. "
            "Get your FLOCK_API_KEY from https://train.flock.io/flock_api")
    if os.environ.get("HG_USERNAME") is None:
        raise Exception("HG_USERNAME not found in environment variables.")
    TASK_ID = 2
    node = TrainNode(TASK_ID)
    node.train()
    node.push()
