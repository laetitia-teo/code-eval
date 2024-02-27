import gc
import torch


def get_loss(model, input_ids, attention_mask, loss_attention_mask=None, 
                      loss_fct=torch.nn.CrossEntropyLoss(reduce=False)):
    batch_size, seq_len = input_ids.shape
    labels = input_ids.clone()
    # print(input_ids.shape)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    shift_logits = shift_logits.view(-1, model.config.vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    # average non-masked tokens over seq dim
    loss = loss.view(batch_size, seq_len - 1)
    if loss_attention_mask is None:
        loss = (loss * attention_mask[..., :-1].contiguous()).sum(-1) / attention_mask.sum(-1)
    else:
        loss = (loss * loss_attention_mask[..., :-1].contiguous()).sum(-1) / loss_attention_mask.sum(-1)

    return loss


@torch.no_grad()
def get_puzzle_solution_likelihoods():  # TODO complete this
    return 0.


@torch.no_grad()
def get_solution_losses(tokenized_puzzle_archive, model, batch_size=2):
    if tokenized_puzzle_archive:
        try:
            mask = tokenized_puzzle_archive.loss_attention_mask
            mask_puzzle = True
        except AttributeError:
            mask_puzzle = False
    else:
        mask_puzzle = False

    all_losses = []
    for i in range(0, tokenized_puzzle_archive.input_ids.shape[0], batch_size):
        input_ids = tokenized_puzzle_archive.input_ids[i:i+batch_size].to(model.device)
        attention_mask = tokenized_puzzle_archive.attention_mask[i:i+batch_size].to(model.device)

        if not mask_puzzle:
            # use the loss over both puzzle and solution
            loss = get_loss(model, input_ids, attention_mask)
        else:
            # use loss over solution only
            loss_attention_mask = tokenized_puzzle_archive.loss_attention_mask[i:i+batch_size].to(model.device)
            loss = get_loss(model, input_ids, attention_mask, loss_attention_mask)

        all_losses.append(loss.cpu())
    return torch.cat(all_losses, dim=0)


# optimizer must be not have momentum
def get_compression_progress(tokenized_puzzle, tokenized_puzzle_archive, model, optimizer,
                             original_losses=None, batch_size=2):
    # compute likelihood of solutions before
    if original_losses is None:
        original_losses = get_solution_losses(tokenized_puzzle_archive, model, batch_size=batch_size)

    # step on the current puzzle
    # todo: the memory costs seem to keep increasing here, try to fix
    model.train()
    optimizer.zero_grad()
    tokenized_puzzle.labels = tokenized_puzzle.input_ids.clone()
    loss = model(**tokenized_puzzle).loss
    loss.backward()
    optimizer.step()
    model.eval()

    # clear memory
    del loss
    del tokenized_puzzle.labels
    gc.collect()
    torch.cuda.empty_cache()

    # compute likelihood of solutions after
    final_losses = get_solution_losses(tokenized_puzzle_archive, model, batch_size=batch_size)
    differences = final_losses - original_losses
    return differences
