import torch
import time
import csv
import  operator
import torch.nn as nn
import torch.optim as optim
from model import RumourCLS
from tweetDataset import TweetDataset
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

# train_params
batchsize = 16
numworkers = 10
maxlen = 512
epoch = 20

def trainModel(model, criterion, optimizer, scheduler, train_loader, dev_loader, epochs, topk):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    best_acc = 0
    topkmodel = []
    start = time.time()
    print("--Training Start--")

    for ep in range(epochs):
        model.train()
        print("-- Epoch {} Start--".format(ep))
        for it, batch in enumerate(train_loader):
            input_ids, attn_masks, labels = batch['input_ids'].to(device), batch['attn_masks'].to(device), batch[
                'label'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attn_masks)

            loss = criterion(logits.squeeze(-1), labels.float())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            loss.backward()
            optimizer.step()
            scheduler.step()

            if it % 50 == 0:
                probs = torch.sigmoid(logits.unsqueeze(-1))
                accuracy = ((probs > 0.5).long().squeeze() == labels).float().mean()
                runtime = time.time() - start

                print("Iteration {} of epoch {} complete, Running Time: {:.4f}".format(it, ep, runtime))
                print("Train loss: {:.4f}, Train accuracy: {:.4f}.".format(loss.item(), accuracy))

                start = time.time()

        dev_acc, dev_loss = evaluate(model, criterion, dev_loader)
        print("Epoch {} complete.".format(ep))
        print("Dev loss: {:.4f}, Dev accuracy: {:.4f}".format(dev_loss, dev_acc))

        if dev_acc > best_acc:
            print("Accuracy improved from {:.4f} to {:.4f}".format(best_acc, dev_acc))
            best_acc = dev_acc
            model_info = dict()

            if len(topkmodel) < topk:
                model_info['state'] = model.state_dict()
                model_info['Acc'] = best_acc
                topkmodel.append(model_info)
            elif len(topkmodel) == topk:
                smallerAcc = float("inf")
                smallerIndex = 0
                for i in range(len(topkmodel)):
                    if topkmodel[i]['Acc'] < smallerAcc:
                        smallerAcc = topkmodel[i]['Acc']
                        smallerIndex = i
                topkmodel[smallerIndex]['state'] = model.state_dict()
                topkmodel[smallerIndex]['Acc'] = best_acc
        print("-- Epoch {} End--\n".format(ep))

    for i in range(len(topkmodel)):
        stateinfor = topkmodel[i]['state']
        print(topkmodel[i]['Acc'])
        torch.save(stateinfor, 'bert_cls50_{}.pth'.format(i))
    print("--Training End--")

def evaluate(model, criterion, dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    total_acc, total_loss = 0, 0

    with torch.no_grad():
        for it, batch in enumerate(dataloader):
            input_ids, attn_masks, labels = batch['input_ids'].to(device), batch['attn_masks'].to(device), batch[
                'label'].to(device)
            logits = model(input_ids, attn_masks)
            total_loss += criterion(logits.squeeze(-1), labels.float()).item()

            probs = torch.sigmoid(logits.unsqueeze(-1))
            total_acc += (((probs > 0.5).long().squeeze() == labels).float().mean())

    return total_acc / len(dataloader), total_loss / len(dataloader)

def predict(model, test_loader, filename):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model = model.eval()
    predictions_lst = []

    print("--Prediction Start--")

    with torch.no_grad():
        for it, batch in enumerate(test_loader):
            input_ids, attn_masks = batch['input_ids'].to(device), batch['attn_masks'].to(device)
            logits = model(input_ids, attn_masks)
            probs = torch.sigmoid(logits.unsqueeze(-1))
            soft_prob = (probs > 0.5).long()

            for i in soft_prob.squeeze():
                predictions_lst.append(i.item())

    with open(filename, "w", newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Id', 'Predicted'])
        for i in range(len(predictions_lst)):
            writer.writerow([i, predictions_lst[i]])

    print("--Prediction End--")

def main():
    # load data
    traindata = list(open("train.data.txt", 'r').read().split())
    trainlabel = list(open("train.label.txt", 'r').read().split())
    train_set = TweetDataset(data_file=traindata, label_file=trainlabel, max_len=maxlen, tag='train')

    devdata = list(open("dev.data.txt", 'r').read().split())
    devlabel = list(open("dev.label.txt", 'r').read().split())
    dev_set = TweetDataset(data_file=devdata, label_file=devlabel, max_len=maxlen, tag='dev')

    train_loader = DataLoader(dataset=train_set, batch_size=batchsize, shuffle=True, num_workers=numworkers,
                              collate_fn=train_set.my_collate)
    dev_loader = DataLoader(dataset=dev_set, batch_size=batchsize, shuffle=True, num_workers=numworkers,
                            collate_fn=dev_set.my_collate)

    # model_params
    total_steps = len(train_loader) * epoch
    warm_ratio = 0.1
    model = RumourCLS()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-5, eps=1e-8, weight_decay=1e-3)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps= warm_ratio * total_steps, num_training_steps=total_steps)

    # train
    trainModel(model, criterion, optimizer, scheduler, train_loader, dev_loader, epoch, 3)

    testdata = list(open("test.data.txt", 'r').read().split())
    test_set = TweetDataset(data_file=testdata, label_file=None, max_len=maxlen, tag='test')
    test_loader = DataLoader(dataset=test_set, batch_size=batchsize, shuffle=True,  num_workers=numworkers, collate_fn=test_set.my_collate)
    predict(model, test_loader, 'result2.csv')

if __name__ == "__main__":
    main()
