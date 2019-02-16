import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

def train(model, model_name, epoch = 30, early_stopping = 6, loss, 
          optimizer, dataset_train_loader, dataset_validate_loader, 
          mode = "ARC", mode = mode, model2 = model2, en_type = entype):
    '''
    model should always be the ARC models and model2 is for the enhancement models.
    '''
    counter = 0
    l = 0
    l_pre = 0

    for i in range(epoch):
        l_pre = l
    
        if(mode == "ARC"):
            model.train()
        elif(mode == "Enhancement"):
            model.eval()
            model2.train()
        train(i, model, dataset_train_loader, optimizer, loss, mode = mode, model2 = model2, en_type = entype)
        
        if(mode == "ARC"):
            model.eval()
        elif(mode == "Enhancement"):
            model2.eval()
        l = validate(model, dataset_validate_loader, loss, mode = mode, model2 = model2, en_type = entype)
    
        if(np.abs(l - l_pre) <= 0.00001):
            counter +=1
        if(counter >= early_stopping):
            break

    save_model(model, model_name)
    
def train_epoch(epoch, model, dataset_loader, optimizer, loss, mode = "ARC", model2 = None, en_type = None):
    loss_sum = 0
    loss_avg = 0
    
    if(mode == "ARC"):
        model.train()
    elif(mode == "Enhancement"):
        model.eval()
        model2.train()
        
    for i, (features, labels) in enumerate(tqdm.tqdm(dataset_loader)):
        features = features.cuda()
        labels = labels.cuda()
        
        if(mode == "ARC"):
            # Train ARC
            optimizer.zero_grad()
            out = model(features)
            l = loss(out, labels)
            l.backward()
            optimizer.step()
            
        elif(mode == "Enhancement"):
            # Train Enhancement
            optimizer.zero_grad()
            with torch.no_grad():    
                out_arc = model(features)
                
            if(en_type == "voice" ):
                out_en = model2(out_arc[:,:1025,:])
                l = loss(out_en, labels[:,:1025,:])
            elif(en_type == "others"):
                out_en = model2(out_arc[:,1025:,:])
                l = loss(out_en, labels[:,1025:,:])
        
            l.backward()
            optimizer.step()
        
        loss_sum += l.item()
        if ( i % 100 == 99):
                loss_avg = loss_sum/100
                print("Epoch: " + str(epoch) + " ARC Loss: " + str(loss_avg))
                loss_sum = 0

def validate_epoch(model, dataset_loader, loss, mode = "ARC", model2 = None, en_type = None):
    # TODO: parallelize
    loss_sum = 0
    loss_avg = 0
    counter = 0
    hop = 256
    
    model.eval()
    if(model2 is not None):
        model2.eval()
        
    with torch.no_grad():
        for i, (features, labels) in enumerate(tqdm.tqdm(dataset_loader)):
            features = features.cuda()
            labels = labels.cuda()
            # Predict based on the original seting of input size
            for ii in range(0, features.shape[2], hop):
                if(ii + hop >= features.shape[2]):
                    out_arc = model(features[:, :, -hop:])
                    label = labels[:, :, -hop:]
                else:
                    out_arc = model(features[:, :, ii:ii + hop])
                    label = labels[:, :, ii:ii + hop]
                    
                if(mode == "Enhancement"):
                    if(en_type == "voice"):
                        out_en = model2(out_arc[:,:1025,:])
                        l = loss(out_en, label[:,:1025,:])
                    elif(en_type == "others"):
                        out_en = model2(out_arc[:,1025:,:])
                        l = loss(out_en, label[:,1025:,:])
                elif(mode == "ARC"):
                    l = loss(out_arc, label)
    
                loss_sum += l.item()
                counter += 1
            
    loss_avg = loss_sum/counter
    print("Validation Loss: " + str(loss_avg))
    
    return loss_avg