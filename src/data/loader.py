from torch.utils.data import DataLoader

def create_dataloader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=None
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )