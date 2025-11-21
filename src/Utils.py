import torch
import time
import matplotlib.pyplot as plt
import os
import pandas as pd 
import ast

def batch_accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).sum().item(), y.size(0)


def train(train_loader, val_loader, model, opt, lossfunc, config, overwrite_epoch_print=False, show_batch_time=False, show_plot = True):
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
            lossfunc.reduction = 'sum'
            loss = lossfunc(logits, yb)
            running_loss += loss.item()
            loss.backward()
            opt.step()


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

        #Record train loss/acc
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        #Time stuff
        train_batch_time_cum = batch_time_cum
        train_batch_time_ave = batch_time_cum / curr_batch


        #Validation
        val_loss, val_acc, val_time, _ = test_routine(val_loader, model, lossfunc, config, show_batch_time)  
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        message = f"Epoch {epoch+1}/{config.epochs} -  val loss: {val_loss:.4f} |  val acc: {val_acc:.2f} | train loss {epoch_loss:.4f} | train acc {epoch_acc:.4f} | time {(train_batch_time_cum+val_time)/1000.0:.2f}s | per batch {train_batch_time_ave:.2f}ms"
        if(overwrite_epoch_print):
            print(message, end='\r')
        else:
            print(message)


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
        raise ValueError("mode must be 'Training', 'Validation'")
    

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


def test_routine(test_loader, model, lossfunc, config, show_batch_time=False):
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
            lossfunc.reduction = 'sum'
            loss = lossfunc(logits, target)
            running_loss += loss.item()
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


    avg_loss = running_loss / total
    avg_accuracy = correct / total
    return avg_loss, avg_accuracy, batch_time_cum, batch_time_cum / num_batches

def test(test_loader, model, lossfunc, config, overwrite_epoch_print=False, show_batch_time=False):
    avg_loss, avg_accuracy, total_time, per_batch_time = test_routine(test_loader, model, lossfunc, config, show_batch_time)
    message = f"test loss {avg_loss:.4f} | test acc {avg_accuracy:.4f} | time {total_time/1000.0:.2f}s | per batch {per_batch_time:.2f}ms"
    if(overwrite_epoch_print):
        print(message, end='\r')
    else:
        print(message)


# from https://github.com/mdeff/fma/blob/master/utils.py
def load(filepath):

    filename = os.path.basename(filepath)

    if 'features' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'echonest' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'genres' in filename:
        return pd.read_csv(filepath, index_col=0)

    if 'tracks' in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                   ('track', 'genres'), ('track', 'genres_all')]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                   ('album', 'date_created'), ('album', 'date_released'),
                   ('artist', 'date_created'), ('artist', 'active_year_begin'),
                   ('artist', 'active_year_end')]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ('small', 'medium', 'large')
        try:
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                    'category', categories=SUBSETS, ordered=True)
        except (ValueError, TypeError):
            # the categories and ordered arguments were removed in pandas 0.25
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                     pd.CategoricalDtype(categories=SUBSETS, ordered=True))

        COLUMNS = [('track', 'genre_top'), ('track', 'license'),
                   ('album', 'type'), ('album', 'information'),
                   ('artist', 'bio')]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype('category')

        return tracks



