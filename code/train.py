import torch
import json
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np

from models import Probe
from dataset import create_df, get_gold_data, get_bert_embedding_dict, get_visual_bert_embedding_dict, get_lists_and_dicts

#device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu"

models = ['bert', 'visualbert']

with open('./code/config.json', 'r') as f:
    hyps = json.load(f)
    bert_hyperparameters = hyps[0]
    visual_bert_hyperparameters = hyps[1]


df = create_df('./data/affordance_annotations.txt')

unique_objects, unique_affordances, word_to_index, index_to_word = get_lists_and_dicts(df)

train_pairs = get_gold_data(df[:42])
val_pairs = get_gold_data(df[42:52])
test_pairs = get_gold_data(df[52:])

bert_word_to_embedding = get_bert_embedding_dict([train_pairs + val_pairs + test_pairs])
visual_bert_word_to_embedding = get_visual_bert_embedding_dict([train_pairs + val_pairs + test_pairs])

# The structure in each batch is: 
# 0 bert object embedding, 
# 1 bert affordance embedding, 
# 2 visualbert object embedding, 
# 3 visualbert affordance embedding, 
# 4 truth values,
# 5 object id
# 6 affordance id

def train(model, criterion, optimizer, train_dataloader, val_dataloader, hyperparameters):

    torch.nn.init.uniform_(model.fc1.weight, a=0, b=1)

    model.train()

    epoch_list = []
    val_loss_list = []
    train_loss_list = []
    total_loss = 0

    train_accuracy_list = []
    val_accuracy_list = []

    for epoch in range(hyperparameters['epochs']):
        
        # TRAIN LOOP
        training_loss = 0
        epoch_accuracy = 0
        
        for _, batch in enumerate(train_dataloader):
            
            obj = batch[0]
            affordance = batch[1]
            truth_value = batch[2]
            
            output = model(obj, affordance)
            loss = criterion(output,truth_value)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            training_loss += loss.item()
            
            # calculate training accuracy
            prediction = torch.argmax(output, dim=1)
            correct_predictions = torch.eq(prediction,truth_value).long()
            batch_accuracy = float(sum(correct_predictions)/len(correct_predictions))
            epoch_accuracy += batch_accuracy
        
        # VALIDATION LOOP

        validation_loss = 0
        model.eval()
        
        val_epoch_accuracy = 0
        
        for _, batch in enumerate(val_dataloader):
            
            obj = batch[0]
            affordance = batch[1]
            truth_value = batch[2]
            
            output = model(obj, affordance)
            loss = criterion(output,truth_value)
            validation_loss += loss.item()
            
            # calculate validation accuracy
            prediction = torch.argmax(output, dim=1)
            correct_predictions = torch.eq(prediction,truth_value).long()
            batch_accuracy = float(sum(correct_predictions)/len(correct_predictions))
            val_epoch_accuracy += batch_accuracy
        
        epoch_list.append(epoch+1)
        training_loss_avg = training_loss/len(train_dataloader)
        train_loss_list.append(training_loss_avg)
        validation_loss_avg = validation_loss/len(val_dataloader)
        val_loss_list.append(validation_loss_avg)
        
        best_accuracy = max(val_accuracy_list) if val_accuracy_list else 0

        if (val_epoch_accuracy/len(val_dataloader)) >= best_accuracy:
            torch.save(model.state_dict(), "|".join([f"{k}_{v}" for k, v in hyperparameters.items()]))

        train_accuracy_list.append(epoch_accuracy/len(train_dataloader))
        val_accuracy_list.append(val_epoch_accuracy/len(val_dataloader))

        if (epoch+1) % 100 == 0 and epoch != 0:
            print("Epoch: {}".format(epoch+1))
            print("Training loss: {}".format(training_loss_avg))
            print("Validation loss: {}".format(validation_loss_avg))
            print("Training accuracy: {} %".format(np.round((epoch_accuracy/len(train_dataloader)) * 100, 2)))
            print("Validation accuracy: {} %".format(np.round((val_epoch_accuracy/len(val_dataloader)) * 100, 2)))

    return val_accuracy_list.index(best_accuracy)+1, best_accuracy

def main():

    for model_name in models:

        if model_name == 'bert':
            
            torch.manual_seed(0)

            bert_probe = Probe().to(device)

            criterion = nn.NLLLoss()

            optimizer = optim.Adam(
            bert_probe.parameters(),
            lr=bert_hyperparameters["learning_rate"])

            train_data = [(bert_word_to_embedding[x], bert_word_to_embedding[y], z, word_to_index[x], word_to_index[y]) for x,y,z in train_pairs]
            val_data = [(bert_word_to_embedding[x], bert_word_to_embedding[y], z, word_to_index[x], word_to_index[y]) for x,y,z in val_pairs]

            train_dataloader = DataLoader(train_data, batch_size=bert_hyperparameters["batch_size"], shuffle=True)
            val_dataloader = DataLoader(val_data, batch_size=bert_hyperparameters["batch_size"], shuffle=True)

            print(f'Start training BERT Probe')
            best_epoch, best_accuracy = train(bert_probe, criterion, optimizer, train_dataloader, val_dataloader, bert_hyperparameters)
            print(f'BERT Probe saved at epoch {best_epoch} with validation accuracy {np.round(best_accuracy * 100, 2)} %')

        elif model_name == 'visualbert':

            torch.manual_seed(6)

            visual_bert_probe = Probe().to(device)

            criterion = nn.NLLLoss()

            optimizer = optim.Adam(
            visual_bert_probe.parameters(),
            lr=visual_bert_hyperparameters["learning_rate"])

            train_data = [(visual_bert_word_to_embedding[x], visual_bert_word_to_embedding[y], z, word_to_index[x], word_to_index[y]) for x,y,z in train_pairs]
            val_data = [(visual_bert_word_to_embedding[x], visual_bert_word_to_embedding[y], z, word_to_index[x], word_to_index[y]) for x,y,z in val_pairs]

            train_dataloader = DataLoader(train_data, batch_size=bert_hyperparameters["batch_size"], shuffle=True)
            val_dataloader = DataLoader(val_data, batch_size=bert_hyperparameters["batch_size"], shuffle=True)
            
            print(f'Start training VisualBERT Probe')
            best_epoch, best_accuracy = train(visual_bert_probe, criterion, optimizer, train_dataloader, val_dataloader, visual_bert_hyperparameters)
            print(f'VisualBERT Probe saved at epoch {best_epoch} with validation accuracy {np.round(best_accuracy * 100, 2)} %')
    return

if __name__ == '__main__':
    
    main()