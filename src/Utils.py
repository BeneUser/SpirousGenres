import torch

def batch_accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).sum().item(), y.size(0)


def train(train_dataset, train_loader, model, opt, lossfunc, config):
    model.train()
    for epoch in range(config.epochs):
        running_loss = 0.0
        correct = 0.0
        total = 0.0
        for xb, yb in train_loader:
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

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct / total
        print(f"epoch {epoch+1} | train loss {epoch_loss:.4f} | train acc {epoch_acc:.4f}")

def test(test_dataset, test_loader, model, lossfunc, config):
    model.eval()
    running_loss = 0.0
    correct = 0.0
    total = 0.0
    for data, target in test_loader:
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

    loss = running_loss / len(test_dataset) #MAYBE UNCORRECT?????
    accuracy = correct / total
    print(f"test loss {loss:.4f} | test acc {accuracy:.4f}")