import torch
import time

def batch_accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).sum().item(), y.size(0)


def train(train_dataset, train_loader, model, opt, lossfunc, config, show_batch_time=False):
    model.train()
    for epoch in range(config.epochs):
        running_loss = 0.0
        correct = 0.0
        total = 0.0

        #Timing stuff
        num_batches = len(train_loader)
        curr_batch = 0
        batch_time_cum = 0
        for xb, yb in train_loader:
            batch_start = time.time_ns()

            if xb.dim() == 2:    
                xb = xb.unsqueeze(1)
            if yb.dtype != torch.long:
                yb = yb.long()

            xb, yb = xb.to(config.device), yb.to(config.device)

            opt.zero_grad()
            logits = model(xb)
            loss = lossfunc(logits, yb)
            loss.backward()
            opt.step()

            running_loss += loss.item() * xb.size(0)
            c, t = batch_accuracy(logits, yb)
            correct += c
            total += t

            #Print batch time in ms
            curr_batch += 1
            batch_end = time.time_ns()
            batch_time = (batch_end - batch_start) / 1000000 #ms
            batch_time_cum += batch_time
            if(show_batch_time):
               print(f"({curr_batch}/{num_batches}) batch time {batch_time:.2f}ms | cumulative {batch_time_cum:.2f}ms | average {batch_time_cum / curr_batch:.2f}ms", end="\r")

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct / total
        print(f"epoch {epoch+1} | train loss {epoch_loss:.4f} | train acc {epoch_acc:.4f} | time {batch_time_cum/1000.0:.2f}s | per batch {batch_time_cum / num_batches:.2f}ms")

def test(test_dataset, test_loader, model, lossfunc, config, show_batch_time=False):
    model.eval()
    running_loss = 0.0
    correct = 0.0
    total = 0.0

    #Timing stuff
    num_batches = len(test_loader)
    curr_batch = 0
    batch_time_cum = 0
    for data, target in test_loader:
        batch_start = time.time_ns()
        if data.dim() == 2:    
            data = data.unsqueeze(1)
        if target.dtype != torch.long:
            target = target.long()

        
        # Move data to GPU (if available)
        data, target = data.to(config.device), target.to(config.device) 
        with torch.no_grad():
            logits = model(data)

            #Loss
            running_loss += lossfunc(logits, target)
            #Accuracy
            c, t = batch_accuracy(logits, target)
            correct += c
            total += t
            
        #Print batch time in ms
        curr_batch += 1
        batch_end = time.time_ns()
        batch_time = (batch_end - batch_start) / 1000000 #ms
        batch_time_cum += batch_time
        if(show_batch_time):
            print(f"({curr_batch}/{num_batches}) batch time {batch_time:.2f}ms | cumulative {batch_time_cum:.2f}ms | average {batch_time_cum / curr_batch:.2f}ms", end="\r")


    loss = running_loss / len(test_dataset) #MAYBE UNCORRECT?????
    accuracy = correct / total
    print(f"test loss {loss:.4f} | test acc {accuracy:.4f} | time {batch_time_cum/1000.0:.2f}s | per batch {batch_time_cum / num_batches:.2f}ms")