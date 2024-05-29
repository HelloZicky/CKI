import torchvision
import torch
import os
import copy
from sklearn import metrics
import numpy as np

class UnitingModelForVision(torch.nn.Module):
    def __init__(self, model_type):
        super().__init__()
        self.model_type = model_type
        self.a = 0.4
        self.c = 500
        if model_type == "resnet34":
            self.fc_assess = torch.nn.Linear(512, 512)
            self.dropout = torch.nn.Dropout(0.8)
            self.weight_key = 'fc.weight'
            self.bias_key = 'fc.bias'
        elif model_type == "mobilenetv3":
            self.classifier_0_access = torch.nn.Linear(576, 576)
            self.classifier_3_access = torch.nn.Linear(1024, 1024)
            self.dropout = torch.nn.Dropout(0.8)
            self.weight_0_key = 'classifier.0.weight'
            self.weight_3_key = 'classifier.3.weight'
            self.bias_0_key = 'classifier.0.bias'
            self.bias_3_key = 'classifier.3.bias'
            self.weight_keys = [self.weight_0_key, self.weight_3_key]
            self.bias_keys = [self.bias_0_key, self.bias_3_key]
            self.modules = [self.classifier_0_access, self.classifier_3_access]
        else:
            raise NotImplementedError
        
    def forward(self, x, model, graft_model):
        def entropy(x, n=10):
            x = x.reshape(-1)
            scale = (x.max() - x.min()) / n
            entropy = torch.tensor(0.).to(x.device)
            for i in range(n):
                p = torch.sum((x >= x.min() + i * scale) * (x < x.min() + (i + 1) * scale), dtype=torch.float) / len(x)
                if p != 0:
                    entropy -= p * torch.log(p)
            return entropy
        
        if self.model_type == 'resnet34':
            x = self.model_forward(x, model)
            param_i = model.state_dict()[self.weight_key]
            param_j = graft_model.state_dict()[self.weight_key]
            sigmoid = torch.nn.Sigmoid()
            param_direction = torch.abs(param_i - param_j) * 10
            param_direction = self.fc_assess(param_direction)
            w_local = sigmoid(param_direction)
            # w_local = param_direction
            w_global = self.a / torch.pi * torch.arctan(self.c * (entropy(param_i) - entropy(param_j))) + 0.5
            w = w_global * w_local + (1 - w_global)
            
            param_uniting = param_i * w + param_j * (1 - w)

            # x = self.dropout(x)
            x = torch.matmul(x, param_uniting.transpose(0, 1)) + model.state_dict()[self.bias_key]

            return x
        
        elif self.model_type == 'mobilenetv3':
            x = self.model_forward(x, model)
            x = model.classifier[0](x)
            x = model.classifier[1](x)
            x = model.classifier[2](x)
            for i in range(1,len(self.weight_keys)):
                param_i = model.state_dict()[self.weight_keys[i]]
                param_j = graft_model.state_dict()[self.weight_keys[i]]
                sigmoid = torch.nn.Sigmoid()
                param_direction = torch.abs(param_i - param_j) * 10
                param_direction = self.modules[i](param_direction)
                w_local = sigmoid(param_direction)
                w_global = self.a / torch.pi * torch.arctan(self.c * (entropy(param_i) - entropy(param_j))) + 0.5
                w = w_global * w_local + (1 - w_global)
            
                param_uniting = param_i * w + param_j * (1 - w)

                # x = self.dropout(x)
                x = torch.matmul(x, param_uniting.transpose(0, 1)) + model.state_dict()[self.bias_keys[i]]
                
                if i + 1 != len(self.weight_keys):
                    x = torch.tanh(x)
                else:
                    return x

        else:
            raise NotImplementedError
        
    def dump_model(self, model, graft_model):
        new_model = copy.deepcopy(model)
        def entropy(x, n=10):
            x = x.reshape(-1)
            scale = (x.max() - x.min()) / n
            entropy = torch.tensor(0.).to(x.device)
            for i in range(n):
                p = torch.sum((x >= x.min() + i * scale) * (x < x.min() + (i + 1) * scale), dtype=torch.float) / len(x)
                if p != 0:
                    entropy -= p * torch.log(p)
            return entropy
        
        if self.model_type == 'resnet34':
            with torch.no_grad():
                
                new_param = copy.deepcopy(model.state_dict())
                
                param_i = model.state_dict()[self.weight_key]
                param_j = graft_model.state_dict()[self.weight_key]
                sigmoid = torch.nn.Sigmoid()
                param_direction = torch.abs(param_i - param_j) * 10
                param_direction = self.fc_assess(param_direction)
                w_local = sigmoid(param_direction)
                # w_local = param_direction
                w_global = self.a / torch.pi * torch.arctan(self.c * (entropy(param_i) - entropy(param_j))) + 0.5
                w = w_global * w_local + (1 - w_global)
                param_uniting = param_i * w + param_j * (1 - w)
                
                new_param['fc.weight'] = param_uniting
                new_model.load_state_dict(new_param)
                return new_model
        elif self.model_type == 'mobilenetv3':
            with torch.no_grad():
                new_param = copy.deepcopy(model.state_dict())
                
                for i in range(1,len(self.weight_keys)):
                    param_i = model.state_dict()[self.weight_keys[i]]
                    param_j = graft_model.state_dict()[self.weight_keys[i]]
                    sigmoid = torch.nn.Sigmoid()
                    param_direction = torch.abs(param_i - param_j)
                    param_direction = self.modules[i](param_direction)
                    w_local = sigmoid(param_direction)
                    w_global = self.a / torch.pi * torch.arctan(self.c * (entropy(param_i) - entropy(param_j))) + 0.5
                    w = w_global * w_local + (1 - w_global)
                
                    param_uniting = param_i * w + param_j * (1 - w)

                    new_param[self.weight_keys[i]] = param_uniting
                    
                    if i+1 == len(self.weight_keys):
                        new_model.load_state_dict(new_param)
                        return new_model
                
        else:
            raise NotImplementedError
        
    def model_forward(self, x, model):
        if self.model_type == "resnet34":
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)

            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)

            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            # x = model.fc(x)
            
            return x
        elif self.model_type == "mobilenetv3":
            x = model.features(x)

            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            
            # x = model.classifier(x)
            
            return x
        else:
            raise NotImplementedError
    
    
def main():
    torch.manual_seed(1)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_dataset = torchvision.datasets.CIFAR100('/path/to/your/CIFAR100', train=True, transform=torchvision.transforms.ToTensor())
    test_dataset = torchvision.datasets.CIFAR100('/path/to/your/CIFAR100', train=False, transform=torchvision.transforms.ToTensor())
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    model = torchvision.models.mobilenetv3.mobilenet_v3_small(num_classes=100)
    model.load_state_dict(torch.load('/path/to/your/trained/model/1'))
    model.to(device)
    
    graft_model = torchvision.models.mobilenetv3.mobilenet_v3_small(num_classes=100)
    graft_model.load_state_dict(torch.load('/path/to/your/trained/model/2'))
    graft_model.to(device)

    uniting = UnitingModelForVision('mobilenetv3').to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(uniting.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)
    
    best_acc = 0
    for epoch in range(25):
        uniting.train()
        total_loss = 0
        for batch, (x, y) in enumerate(train_dataloader):
            x, y = x.to(device), y.to(device)
            
            pred = uniting(x, model, graft_model)
            loss = criterion(pred, y)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if batch % 200 == 0:
                print('Epoch ', epoch, ', step ', batch, ': loss = ', total_loss, sep='')
                total_loss = 0
            else:
                total_loss += loss.item()
        scheduler.step()
        
        with torch.no_grad():
            new_model = uniting.dump_model(model, graft_model).to(device)
            size = len(test_dataloader.dataset)
            num_batches = len(test_dataloader)
            new_model.eval()
            pred_list = []
            prob_list = []
            labl_list = []
            y_list = []
            correct = 0
            test_loss = 0
            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)
                pred = new_model(X)
                prob = torch.softmax(pred, dim=1)
                
                y_list.extend(y.cpu().numpy())
                pred_list.extend(pred.cpu().numpy())
                prob_list.extend(prob.cpu().numpy())
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                test_loss += criterion(pred, y).item()
                
                _, predicted = torch.topk(pred, k=3, dim=1)
                labl_list.extend(predicted.cpu().numpy())
                
        
        acc = correct / size
        auc = metrics.roc_auc_score(np.array(y_list), np.array(prob_list), multi_class='ovr')
        logloss = metrics.log_loss(np.array(y_list), pred_list)
        
        top3 = [label[:3] for label in labl_list]
        acc3 = metrics.top_k_accuracy_score(y_list, pred_list, k=3)
        labels = np.eye(100)[np.array(labl_list)]
        # acc3 = metrics.top_k_categorical_accuracy(torch.tensor(true_labels), torch.tensor(predictions), k=3)
        # precision_at_3 = metrics.precision_score(y_list, labl_list, average='macro', k=3)
        # recall3 = metrics.top_k_accuracy_score(y_list, pred_list, k=3)
        # acc3 = recall3
        entropyloss = test_loss / num_batches
        
        
        print('='*50)
        print(f'Epoch {epoch} eval: acc = {acc*100:>.4f}%, auc = {auc:>.6f}, entropyloss = {entropyloss}, logloss = {logloss:>.6f}, acc3 = {acc3*100:>.4f}%')
        if acc > best_acc:
            best_acc = acc
            print('New Best acc:', acc)
            best_acc_uniting = copy.deepcopy(uniting)
            best_acc_model = copy.deepcopy(new_model)
        print('='*50)
        
    output_dir = os.path.join('graft', 'mobilenetv3s_v6', 'CIFAR100')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    torch.save(best_acc_uniting.state_dict(), os.path.join(output_dir, 'uniting.pth'))
    torch.save(best_acc_model.state_dict(), os.path.join(output_dir, 'graft.pth'))

if __name__ == "__main__":
    main()