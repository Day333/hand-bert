import numpy as np
import torch
import torch.nn as nn
import time
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertModel



class BertClassifier(nn.Module):
    def __init__(self, path):
        super(BertClassifier, self).__init__()
        D_in, H, D_out = 768, 100, 2

        self.bert = BertModel.from_pretrained(path)

        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, D_out)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        # 为分类任务提取标记[CLS]的最后隐藏状态，因为要连接传到全连接层去
        last_hidden_state_cls = outputs[0][:, 0, :]

        logits = self.classifier(last_hidden_state_cls)

        return logits


def initialize_model(device, train_dataloader, bert_path, epochs=10):
    bert_classifier = BertClassifier(bert_path)
    bert_classifier.to(device)
    optimizer = AdamW(bert_classifier.parameters(), lr=5e-5, eps=1e-8)

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value
                                                num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler


def train(model, device, optimizer, scheduler, train_dataloader, test_dataloader=None, epochs=10, evaluation=False):
    for epoch_i in range(epochs):

        print(f"{'Epoch':^7} | {'per 10 epoch Batch':^9} | {'train Loss':^12} | {'test Loss':^10} | {'train acc':^9} | {'time':^9}")
        print("-" * 80)

        t0_epoch, t0_batch = time.time(), time.time()

        total_loss, batch_loss, batch_counts = 0, 0, 0
        loss_fn = nn.CrossEntropyLoss()

        model.train()

        for step, batch in enumerate(train_dataloader):
            batch_counts += 1

            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            model.zero_grad()

            logits = model(b_input_ids, b_attn_mask)


            loss = loss_fn(logits, b_labels)

            batch_loss += loss.item()
            total_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            if (step % 10 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                time_elapsed = time.time() - t0_batch

                print(
                    f"{epoch_i + 1:^7} | {step:^10} | {batch_loss / batch_counts:^14.6f} | {'-':^12} | {'-':^13} | {time_elapsed:^9.2f}")

                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        avg_train_loss = total_loss / len(train_dataloader)

        print("-" * 80)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation:

            test_loss, test_accuracy = evaluate(model, test_dataloader, loss_fn, device)
            time_elapsed = time.time() - t0_epoch

            print(
                f"{epoch_i + 1:^7} | {'-':^10} | {avg_train_loss:^14.6f} | {test_loss:^12.6f} | {test_accuracy:^12.2f}% | {time_elapsed:^9.2f}")
            print("-" * 80)
        print("\n")


def evaluate(model, test_dataloader, loss_fn, device):

    model.eval()

    test_accuracy = []
    test_loss = []

    for batch in test_dataloader:
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # 计算误差
        loss = loss_fn(logits, b_labels.long())
        test_loss.append(loss.item())

        preds = torch.argmax(logits, dim=1).flatten()


        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        test_accuracy.append(accuracy)

    val_loss = np.mean(test_loss)
    val_accuracy = np.mean(test_accuracy)

    return val_loss, val_accuracy
