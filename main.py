from node import TrainNode

if __name__ == '__main__':
    TASK_ID = 2
    node = TrainNode(TASK_ID)
    node.train()
    node.push()
