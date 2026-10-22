import torch 

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device
):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        pixel_values = batch["image"].to(device)
        labels = batch["labels"].to(device)

        input_ids = batch.get("input_ids")
        if input_ids is not None:
            input_ids = input_ids.to(device)

        optimizer.zero_grad()

        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels
        )

        loss = outputs["loss"]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)