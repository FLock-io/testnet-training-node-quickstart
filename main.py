from node import TrainNode

if __name__ == '__main__':
    TASK_ID = 2
    node = TrainNode(TASK_ID, FLOCK_API_KEY=FLOCK_API_KEY, HF_USERNAME=HF_USERNAME)
    node.train()
    node.push()
