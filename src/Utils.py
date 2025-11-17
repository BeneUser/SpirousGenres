import torch
import time
import matplotlib.pyplot as plt

def batch_accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).sum().item(), y.size(0)


def train(train_dataset, train_loader, val_dataset, val_loader, model, opt, lossfunc, config, show_batch_time=False, show_plot = True):
    model.train()
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

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

        train_batch_time_cum = batch_time_cum
        train_batch_time_ave = batch_time_cum / curr_batch

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        



        # validation  
        model.eval()
        running_loss = 0
        correct = 0
        total = 0

        #Timing stuff
        num_batches = len(val_loader)
        curr_batch = 0
        batch_time_cum = 0
        for data, target in val_loader:
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
                
            # Print batch time in ms
            curr_batch += 1
            batch_end = time.time_ns()
            batch_time = (batch_end - batch_start) / 1000000 #ms
            batch_time_cum += batch_time
            if(show_batch_time):
                print(f"({curr_batch}/{num_batches}) batch time {batch_time:.2f}ms | cumulative {batch_time_cum:.2f}ms | average {batch_time_cum / curr_batch:.2f}ms", end="\r")


        avg_val_loss = running_loss / total
        val_losses.append(avg_val_loss)
        val_acc = correct / total
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{config.epochs} -  Val Loss: {avg_val_loss:.4f} |  Val Acc: {val_acc:.2f} | train loss {epoch_loss:.4f} | train acc {epoch_acc:.4f} | time {train_batch_time_cum/1000.0:.2f}s | per batch {train_batch_time_ave:.2f}ms")
        #print(f"epoch {epoch+1}/{config.epochs} | train loss {epoch_loss:.4f} | train acc {epoch_acc:.4f} | time {batch_time_cum/1000.0:.2f}s | per batch {batch_time_cum / num_batches:.2f}ms")

    if show_plot == True:
        print_epoch_evolve(train_losses, train_accuracies, config=config, mode="Training")
        print_epoch_evolve(val_losses, val_accuracies, config=config, mode="Validation")

        

def print_epoch_evolve(history_loss, history_acc, config, mode):
    # mode: "Training" or "Validation"

    if mode == "Training":
        acc_label = "Training Accuracy"
        loss_label = "Training Loss"
        color_acc = "tab:blue"
        color_loss = "tab:brown"

    elif mode == "Validation":
        acc_label = "Validation Accuracy"
        loss_label = "Validation Loss"
        color_acc = "tab:orange"
        color_loss = "tab:purple"
    else:
        raise ValueError("mode must be 'Training' or 'Validation'")
    

    fig, ax1 = plt.subplots()
    epochs = range(1, config.epochs + 1)

    # Left y-axis → Training/Validation Loss
    ax1.plot(epochs, history_loss, color=color_loss, label=loss_label)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel(loss_label, color=color_loss)
    ax1.tick_params(axis='y', labelcolor=color_loss)

    # Right y-axis → Training/Validation Accuracy
    ax2 = ax1.twinx()
    ax2.plot(epochs, history_acc, color=color_acc, label=acc_label)
    ax2.set_ylabel(acc_label, color=color_acc)
    ax2.tick_params(axis='y', labelcolor=color_acc)

    # Optional: combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title(f"{loss_label} and {acc_label} over Epochs")
    plt.show()


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