import torch 

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    scaler
):
    model.train()
    total_loss = 0.0

    if len(dataloader) == 0:
        raise ValueError("Training data is empty.")

    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)

        pixel_values = batch["image"].to(device)
        labels = batch["labels"].to(device)
        input_ids = batch["input_ids"].to(device)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=scaler.is_enabled()
        ):
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                labels=labels
            )

            loss = outputs["loss"]

        if not torch.isfinite(loss):
            raise ValueError("Training loss is not finite.")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        del outputs, loss, pixel_values, input_ids, labels

    optimizer.zero_grad(set_to_none=True)

    return total_loss / len(dataloader)