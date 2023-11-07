from TestModel import test_model
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def load_and_standardize_data(path):
    df = pd.read_csv(path, sep=',')
    df = df.fillna(-99)
    df = df.values.reshape(-1, df.shape[1]).astype('float32')
    X_train, X_test = train_test_split(df, test_size=0.3, random_state=42)
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)   
    return X_train, X_test, scaler

class DataBuilder(Dataset):
    def __init__(self, path, train=True):
        self.X_train, self.X_test, self.standardizer = load_and_standardize_data(path)
        if train:
            self.x = torch.from_numpy(self.X_train)
            self.len=self.x.shape[0]
        else:
            self.x = torch.from_numpy(self.X_test)
            self.len=self.x.shape[0]
        del self.X_train
        del self.X_test 
    def __getitem__(self,index):      
        return self.x[index]
    def __len__(self):
        return self.len

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")
    
    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self.mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss_MSE + loss_KLD

class Autoencoder(nn.Module):
    def __init__(self,D_in,H=50,H2=12,latent_dim=3):
        #Encoder
        super(Autoencoder,self).__init__()
        self.linear1=nn.Linear(D_in,H)
        self.lin_bn1 = nn.BatchNorm1d(num_features=H)
        self.linear2=nn.Linear(H,H2)
        self.lin_bn2 = nn.BatchNorm1d(num_features=H2)
        self.linear3=nn.Linear(H2,H2)
        self.lin_bn3 = nn.BatchNorm1d(num_features=H2)
        
        # Latent vectors
        self.fc1 = nn.Linear(H2, latent_dim)
        self.bn1 = nn.BatchNorm1d(num_features=latent_dim)
        self.fc21 = nn.Linear(latent_dim, latent_dim)
        self.fc22 = nn.Linear(latent_dim, latent_dim)

        # Sampling vector
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc_bn3 = nn.BatchNorm1d(latent_dim)
        self.fc4 = nn.Linear(latent_dim, H2)
        self.fc_bn4 = nn.BatchNorm1d(H2)
        
        # Decoder
        self.linear4=nn.Linear(H2,H2)
        self.lin_bn4 = nn.BatchNorm1d(num_features=H2)
        self.linear5=nn.Linear(H2,H)
        self.lin_bn5 = nn.BatchNorm1d(num_features=H)
        self.linear6=nn.Linear(H,D_in)
        self.lin_bn6 = nn.BatchNorm1d(num_features=D_in)
        
        self.relu = nn.ReLU()
        
    def encode(self, x):
        lin1 = self.relu(self.lin_bn1(self.linear1(x)))
        lin2 = self.relu(self.lin_bn2(self.linear2(lin1)))
        lin3 = self.relu(self.lin_bn3(self.linear3(lin2)))

        fc1 = F.relu(self.bn1(self.fc1(lin3)))

        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)
        
        return r1, r2
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3)))

        lin4 = self.relu(self.lin_bn4(self.linear4(fc4)))
        lin5 = self.relu(self.lin_bn5(self.linear5(lin4)))
        return self.lin_bn6(self.linear6(lin5))
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def generate_fake(mu, logvar, no_samples, scaler, model):
    #With trained model, generate some data
    sigma = torch.exp(logvar/2)
    q = torch.distributions.Normal(mu.mean(axis=0), sigma.mean(axis=0))
    z = q.rsample(sample_shape=torch.Size([no_samples]))
    with torch.no_grad():
        pred = model.decode(z).cpu().numpy()
    fake_data = scaler.inverse_transform(pred)
    return fake_data

def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

# When you have all the code in place to generate synthetic data, uncomment the code below to run the model and the tests. 
def main():
    # Get a device and set up data paths. You need paths for the original data, the data with just loan status = 1 and the new augmented dataset.

    device = get_device()
    BATCH_SIZE = 1024
    # Loading original loan data
    original_data = pd.read_csv('data/loan_continuous.csv')
    # Get basic metrics
    print('Original Imbalanced Data Set Metrics.')
    print(original_data.shape)
    print(original_data.head())
    print(original_data.describe())
    
    # Save column labels to list for later
    cols = original_data.columns.values.tolist()
    print('Data Set Columns Names.')
    print(cols)
    
    # Make copy of the dataframe
    my_data = original_data.copy()

    # Select only loans with loan status = 1
    my_data = original_data[original_data['Loan Status'] == 1]
    print('Subset of Data with Loan Status = 1.')
    print(my_data)
    print(my_data.shape)
    
    # Get basic metrics for subset data
    print('Subset Data Set Metrics.')
    print(my_data.shape)
    print(my_data.head())
    print(my_data.describe())
    print(my_data.isnull().sum())
    
    # Save new dataframe with loan status = 1 without index
    my_data.to_csv('data/loan_Status_one.csv', index=False)
    
    # Prepare data for pytorch
    DATA_PATH_SATUS_ONE = 'data/loan_Status_one.csv'
    DATA_PATH_CONTINUOUS = 'data/loan_continuous.csv'

    
    # Create DataLoaders for training and validation 
    training_data_set=DataBuilder(DATA_PATH_SATUS_ONE, train=True)
    test_data_set=DataBuilder(DATA_PATH_CONTINUOUS, train=False)
    training_loader=DataLoader(dataset=training_data_set,batch_size=BATCH_SIZE)
    test_loader=DataLoader(dataset=test_data_set,batch_size=BATCH_SIZE)

    print('Training and Test Loader Data.')
    print(training_loader.dataset.x.shape, test_loader.dataset.x.shape)
    print(training_loader.dataset.x)
    
    print('TestModel of original Imblanced Data Set.')
    #print("Press any key to continue...")
    print('Running TestModel of original Imbalanced Data Set.')
    test_model(DATA_PATH_CONTINUOUS)
    
    print("Train and validate subset data Loan Status = 1")
    #print("Press any key to continue...")
    print("Training and validating subset data Loan Status = 1")    
    # Train and validate the model 
    
    num_epoch = 2000
    LR = 0.001         # learning rate

    # Initiate model
    D_in = training_loader.dataset.x.shape[1]
    H = 50
    H2 = 12
    model = Autoencoder(D_in, H, H2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    

    # Initiate Customloss
    loss_mse = CustomLoss()
    
    training_losses = []
    test_losses = []

    def train(epoch):
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(training_loader):
            data = data.to(device)
            for param in model.parameters():
                param.grad = None
            #optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_mse(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        if epoch % 5 == 0:        
            print('====> Epoch: {} Average training loss: {:.4f}'.format(
                epoch, train_loss / len(training_loader.dataset)))
            training_losses.append(train_loss / len(training_loader.dataset))

    def test(epoch):
        with torch.no_grad():
            test_loss = 0
            for batch_idx, data in enumerate(test_loader):
                data = data.to(device)
                #optimizer.zero_grad()
                for param in model.parameters():
                    param.grad = None
                recon_batch, mu, logvar = model(data)
                loss = loss_mse(recon_batch, data, mu, logvar)
                test_loss += loss.item()
                if epoch % 5 == 0:        
                    print('====> Epoch: {} Average test loss: {:.4f}'.format(
                        epoch, test_loss / len(test_loader.dataset)))
                test_losses.append(test_loss / len(test_loader.dataset))
    for epoch in range(1, num_epoch + 1):
        train(num_epoch)
        test(num_epoch)
   
   
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
    print('Compare original and training data.')
    #print("Press any key to continue...")
    scaler = training_loader.dataset.standardizer
    recon_row = scaler.inverse_transform(recon_batch[0].cpu().numpy().reshape(1, -1))
    real_row = scaler.inverse_transform(test_loader.dataset.x[0].cpu().numpy().reshape(1, -1))
    df = pd.DataFrame(np.stack((recon_row[0], real_row[0])), columns = cols)
    print('Print row of original and training data.')
    print(df.head())

    print('Create augmented data set.')
    #print("Press any key to continue...")
    print('Generating augmented data.')
    data_fake = generate_fake(mu, logvar, 50000, scaler, model)
    data_fake = pd.DataFrame(data_fake)
    data_fake.columns = cols
    data_fake['Loan Status'] = np.round(data_fake['Loan Status']).astype(int)
    data_fake['Loan Status'] = np.where(data_fake['Loan Status']<1, 1, data_fake['Loan Status'])
   
    print('Print augmented data set.')
    print(data_fake.head())
    print(data_fake['Loan Status'].mean())
    print(data_fake.describe())

    #print("Press any key to continue to combine data sets...")
    # Combine the new data with original dataset
    frames = [original_data, data_fake]
    combined_data = pd.concat(frames)
    print('Combined original and augmented data.')
    print(combined_data.head())
    print(original_data.groupby('Loan Status').mean())
    print(combined_data.groupby('Loan Status').mean())
   
    # Save new dataframe without index
    combined_data.to_csv('data/loan_continuous_expanded.csv', index=False)

    print('TestModel on new combined data.')
    # print("Press any key to continue...")
    print('Running TestModel on new combined data.')
    DATA_PATH = 'data/loan_continuous_expanded.csv'
    test_model(DATA_PATH)

if __name__ == '__main__':
    main()
    print("done")