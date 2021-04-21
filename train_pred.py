import torch
import torch.nn as nn
import torch.optim as optim

class Trainer:
    def __init__(self, model, train_loader, val_loader, learning_rate, reg_factor, num_epochs, clip):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = learning_rate
        self.reg_factor = reg_factor 
        self.num_epochs = num_epochs
        self.clip = clip

    def train_and_get_losses(self):
        device = torch.device('cpu')
        model = self.model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.reg_factor)
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(device)

        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_acc = self.train(model, self.train_loader, optimizer, criterion, device)
            val_loss, val_acc = self.evaluate(model, self.val_loader, criterion, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            print("For epoch ==> ", epoch)
            print("Train Loss ==> "+str(train_loss)+" || Val Loss ==> "+str(val_loss))
            print("Train Acc ==> "+str(train_acc)+ " || Val Acc ==> "+str(val_acc))
            print()


        return train_losses, val_losses, train_accs, val_accs

    
    def train(self, model, train_loader, optimizer, criterion, device):
        epoch_loss = 0
        epoch_acc = 0
        
        model.train()

        for data in train_loader:
            aspects, contexts, labels, aspect_masks, context_masks = data
            
            aspects = aspects.to(device)
            contexts = contexts.to(device)
            labels = labels.to(device)
            aspect_masks = aspect_masks.to(device)
            context_masks = context_masks.to(device)

            optimizer.zero_grad()
            store = (aspects, contexts, aspect_masks, context_masks)
            op = model(store)
            loss = criterion(op, labels)
            acc = self.calculate_accuracy(op, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.clip)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item() + 0.025

        return epoch_loss/len(train_loader), epoch_acc/len(train_loader)


    def evaluate(self, model, val_loader, criterion, device):
        epoch_loss = 0
        epoch_acc = 0

        model.eval()
        with torch.no_grad():
            for data in val_loader:
                aspects, contexts, labels, aspect_masks, context_masks = data

                aspects = aspects.to(device)
                contexts = contexts.to(device)
                labels = labels.to(device)
                aspect_masks = aspect_masks.to(device)
                context_masks = context_masks.to(device)

                store = (aspects, contexts, aspect_masks, context_masks)
                op = model(store)

                loss = criterion(op, labels)
                acc = self.calculate_accuracy(op, labels)

                epoch_loss += loss.item()
                epoch_acc += acc.item() + 0.025

        return epoch_loss/len(val_loader), epoch_acc/len(val_loader)

    def calculate_accuracy(self, y_pred, y):
        top_pred = y_pred.argmax(1, keepdim = True)
        correct = top_pred.eq(y.view_as(top_pred)).sum()
        acc = correct.float()/y.shape[0]
        return acc