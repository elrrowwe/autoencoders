import numpy as np


"""
Utility file implementing a simple batching function for loaded torchvision datasets.
"""


def batch(dataset, batch_size: int = 60):
    """
    Function to extract a batch from a torchvision dataset

    :param dataset: the dataset to extract a batch from. needs to have the __get__ method implemented.
    :param batch_size: the size of the batch (60 by default)
    :return: a list containing the items of the batch
    """

    dataset_length = len(dataset)

    batch_list = []

    for i in range(batch_size):
        item = dataset[np.random.randint(0, dataset_length)]
        if item not in batch_list:
            batch_list.append(item[0]) # only appending the image; the __get__ method of the MNIST dataset returns both an image and its label

    return batch_list
