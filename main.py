import torch
from torch.utils.data import DataLoader
from model.BertModel import BDataset, Model
from data.data_process import get_data

def main():
    # Load data
    all_text1, all_text2, all_label = get_data()

    # Read vocabulary mapping
    with open("data/index_2_word.txt", encoding="utf-8") as f:
        index_2_word = f.read().split("\n")
    word_2_index = {word: idx for idx, word in enumerate(index_2_word)}

    # Configuration
    config = {
        "epoch": 100,
        "batch_size": 32,
        "max_len": 128,
        "vocab_size": len(word_2_index),
        "hidden_size": 768,
        "max_position_embeddings": 128,
        "head_num": 4,
        "feed_num": 1024,
        "type_vocab_size": 3,
        "hidden_dropout_prob": 0.2,
        "layer_num": 3,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu"
    }

    # Split data into training and validation sets
    dev_size = 400
    train_text1, train_text2, train_labels = all_text1[:-dev_size], all_text2[:-dev_size], all_label[:-dev_size]
    dev_text1, dev_text2, dev_labels = all_text1[-dev_size:], all_text2[-dev_size:], all_label[-dev_size:]

    # Create datasets and dataloaders
    train_dataset = BDataset(train_text1, train_text2, train_labels, config["max_len"], word_2_index)
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)

    dev_dataset = BDataset(dev_text1, dev_text2, dev_labels, config["max_len"], word_2_index)
    dev_dataloader = DataLoader(dev_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)

    # Initialize model and optimizer
    model = Model(config).to(config["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(config["epoch"]):
        print(f"Epoch {epoch+1}/{config['epoch']}")

        model.train()
        for step, (batch_idx, batch_label, batch_mask_val, batch_seg_idx) in enumerate(train_dataloader):
            batch_idx, batch_label = batch_idx.to(config["device"]), batch_label.to(config["device"])
            batch_mask_val, batch_seg_idx = batch_mask_val.to(config["device"]), batch_seg_idx.to(config["device"])

            # Forward pass and loss computation
            loss = model(batch_idx, batch_seg_idx, batch_mask_val, batch_label)
            loss.backward()

            # Optimization step
            optimizer.step()
            optimizer.zero_grad()

            # Print loss every 200 steps
            if step % 200 == 0:
                print(f"Step {step}, Loss: {loss.item():.2f}")

        # Validation after each epoch
        model.eval()
        mask_correct, mask_total = 0, 0
        next_correct, next_total = 0, 0

        with torch.no_grad():
            for batch_idx, batch_label, batch_mask_val, batch_seg_idx in dev_dataloader:
                batch_idx, batch_label = batch_idx.to(config["device"]), batch_label.to(config["device"])
                batch_mask_val, batch_seg_idx = batch_mask_val.to(config["device"]), batch_seg_idx.to(config["device"])

                # Forward pass for validation
                pre_mask, pre_next = model(batch_idx, batch_seg_idx)

                # Calculate mask accuracy
                mask_correct += (pre_mask[batch_mask_val != 0] == batch_mask_val[batch_mask_val != 0]).sum().item()
                mask_total += (batch_mask_val != 0).sum().item()

                # Calculate next prediction accuracy
                next_correct += (pre_next == batch_label).sum().item()
                next_total += len(batch_label)

        # Compute and print accuracy
        acc_mask = (mask_correct / mask_total) * 100 if mask_total > 0 else 0
        acc_next = (next_correct / next_total) * 100 if next_total > 0 else 0
        print(f"Validation - acc_mask: {acc_mask:.3f}%, acc_next: {acc_next:.3f}%")
        print("*" * 100)

if __name__ == "__main__":
    main()
