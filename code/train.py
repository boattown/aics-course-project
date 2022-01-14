import torch
import json
#import argparse
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from models import Probe
from dataset import create_df, get_gold_data, get_bert_embedding_dict, get_visual_bert_embedding_dict, get_lists_and_dicts

#device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu"

with open('./config.json', 'r') as f:
    hyps = json.load(f)
    bert_hyperparameters = hyps[0]
    visual_bert_hyperparameters = hyps[1]


df = create_df('../data/affordance_annotations.txt')

unique_objects, unique_affordances, word_to_index, index_to_word = get_lists_and_dicts(df)

train_df = df[:42]
val_df = df[42:52]
test_df = df[52:]

train_pairs = get_gold_data(train_df)
val_pairs = get_gold_data(val_df)
test_pairs = get_gold_data(test_df)

bert_word_to_embedding = get_bert_embedding_dict([train_pairs + val_pairs + test_pairs])
visual_bert_word_to_embedding = get_visual_bert_embedding_dict([train_pairs + val_pairs + test_pairs])

train_data = [(bert_word_to_embedding[x], bert_word_to_embedding[y], visual_bert_word_to_embedding[x], visual_bert_word_to_embedding[y], z, word_to_index[x], word_to_index[y]) for x,y,z in train_pairs]
val_data = [(bert_word_to_embedding[x], bert_word_to_embedding[y], visual_bert_word_to_embedding[x], visual_bert_word_to_embedding[y], z, word_to_index[x], word_to_index[y]) for x,y,z in val_pairs]
test_data = [(bert_word_to_embedding[x], bert_word_to_embedding[y], visual_bert_word_to_embedding[x], visual_bert_word_to_embedding[y],z, word_to_index[x], word_to_index[y]) for x,y,z in test_pairs]

train_dataloader = DataLoader(train_data, batch_size=bert_hyperparameters["batch_size"], shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=bert_hyperparameters["batch_size"], shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=bert_hyperparameters["batch_size"], shuffle=True)

# The structure in each batch is: 
# 0 bert object embedding, 
# 1 bert affordance embedding, 
# 2 visualbert object embedding, 
# 3 visualbert affordance embedding, 
# 4 truth values,
# 5 object id
# 6 affordance id

def main():

    print('start training')

    bert_probe = Probe()
    bert_probe.to(device)
    print(bert_probe)
    torch.nn.init.uniform_(bert_probe.fc1.weight, a=0, b=1)
    

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(
        bert_probe.parameters(),
        lr=bert_hyperparameters["learning_rate"]
    )

    epoch_list = []
    val_loss_list = []
    train_loss_list = []
    total_loss = 0

    train_accuracy_list = []
    val_accuracy_list = []

    for epoch in range(bert_hyperparameters["epochs"]):
        
        # TRAIN LOOP
        training_loss = 0
        bert_probe.train()
        
        epoch_accuracy = 0
        
        for i, batch in enumerate(train_dataloader):
            
            obj = batch[0]
            affordance = batch[1]
            truth_value = batch[4]
            
            output = bert_probe(obj, affordance)
            bert_loss = criterion(output,truth_value)
            
            bert_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += bert_loss.item()
            training_loss += bert_loss.item()
            
            # calculate training accuracy
            prediction = torch.argmax(output, dim=1)
            correct_predictions = torch.eq(prediction,truth_value).long()
            batch_accuracy = float(sum(correct_predictions)/len(correct_predictions))
            epoch_accuracy += batch_accuracy
        
        # VALIDATION LOOP
        validation_loss = 0
        bert_probe.eval()
        
        val_epoch_accuracy = 0
        
        for i, batch in enumerate(val_dataloader):
            
            obj = batch[0]
            affordance = batch[1]
            truth_value = batch[4]
            
            output = bert_probe(obj, affordance)
            bert_loss = criterion(output,truth_value)
            validation_loss += bert_loss.item()
            
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
            torch.save(bert_probe.state_dict(), "|".join([f"{k}_{v}" for k, v in bert_hyperparameters.items()]))

        train_accuracy_list.append(epoch_accuracy/len(train_dataloader))
        val_accuracy_list.append(val_epoch_accuracy/len(val_dataloader))

        if (epoch+1) % 100 == 0 and epoch != 0:
            print("Epoch: {}".format(epoch+1))
            print("Training loss: {}".format(training_loss_avg))
            print("Validation loss: {}".format(validation_loss_avg))
            print("Training accuracy: {}".format(epoch_accuracy/len(train_dataloader)))
            print("Validation accuracy: {}".format(val_epoch_accuracy/len(val_dataloader)))
            
    print(f'Model saved at epoch {val_accuracy_list.index(best_accuracy)+1} with validation accuracy {best_accuracy}')
    return

if __name__ == '__main__':
    
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--config_file', type=str, default='./config.json', help='config file with hyperparameters')
    #parser.add_argument('--data_dir', type=str, default='./data/train/', help='directory with the data for training')
    #parser.add_argument('--annotations_file', type=str, default='./train_df.csv', help='file with annotations')
    #arguments = parser.parse_args()
    
    main()