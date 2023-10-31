#Create model training 
import matplotlib.pyplot as plt
import copy
import torch

def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs=20):
    #Get correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    highest_accuracy = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())
    
    def update_model(loader, training):
        current_loss = 0.0
        current_correct = 0
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
                
            with torch.set_grad_enabled(training):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                loss = criterion(outputs, labels)
                
                if(training):
                    loss.backward()
                    optimizer.step()
                    
            current_loss += loss.item() * inputs.size(0)
            current_correct += torch.sum(preds==labels.data) 
        return current_loss, current_correct
    
    for epoch in range(num_epochs):
        print('Epoch: {}'.format(epoch + 1))
        
        #train phase
        model.train()
        train_loss, train_correct = update_model(train_loader, True)        
        scheduler.step()
        
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_accuracy = train_correct.double() / (len(train_loader) * train_loader.batch_size)
        print('Phase: Train  Loss: {} Accuracy: {}'.format(epoch_train_loss, epoch_train_accuracy))
        
        #val phase
        model.eval()
        val_loss, val_correct = update_model(val_loader, False)
        
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_accuracy = val_correct.double() / (len(val_loader) * val_loader.batch_size)
        print('Phase: Validation  Loss: {} Accuracy: {}'.format(epoch_val_loss, epoch_val_accuracy))
        
        if epoch_val_accuracy > highest_accuracy:
            highest_accuracy = epoch_val_accuracy
            best_model_weights = copy.deepcopy(model.state_dict())
    
    print('Training finished. Highest accuracy: {}'.format(highest_accuracy))
    model.load_state_dict(best_model_weights)
    return model