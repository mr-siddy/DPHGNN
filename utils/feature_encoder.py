import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import pandas as pd
import numpy as np
from sklearn import preprocessing

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = ''
def load_data(path):
    df = pd.read_csv(path, sep=',')
    df = df.fillna(-99)
    df_feat = df.iloc[:, :299]
    df_label = df.iloc[:, -1].values
    x = df_feat.values.reshape(-1, df_feat.shape[1]).astype('float32')
    standardizer = preprocessing.StandardScaler()
    x_train = standardizer.fit_transform(x)
    x_train = torch.from_numpy(x_train).to(device)
    return x_train, standardizer, df_label

# dataloader

from torch.utils.data import Dataset, DataLoader

class DataBuilder(Dataset):
    def __init__(self, path):
        self.x, self.standardizer, self.label = load_data(DATA_PATH)
        self.len = self.x.shape[0]
    def __getitem__(self, index):
        return self.x[index]
    def __len__(self):
        return self.len


data_set = DataBuilder(DATA_PATH)
train_loader = DataLoader(dataset = data_set, batch_size=1024)

class Autoencoder(nn.Module):
    def __init__(self, D_in, H=100, H2=50, latent_dim=10):
        super(Autoencoder, self).__init__()
        #Encoder
        self.linear1 = nn.Linear(D_in, H)
        self.lin_bn1 = nn.BatchNorm1d(num_features=H)
        self.linear2 = nn.Linear(H, H2)
        self.lin_bn2 = nn.BatchNorm1d(num_features=H2)
        self.linear3 = nn.Linear(H2, H2)
        self.lin_bn3 = nn.BatchNorm1d(num_features=H2)

        # latent vectors mu and sigma
        self.fc1 = nn.Linear(H2, latent_dim)
        self.fc21 = nn.Linear(latent_dim, latent_dim)
        self.fc22 = nn.Linear(latent_dim, latent_dim)

        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc4 = nn.Linear(latent_dim, H2)
        
        # Decoder
        self.linear4 = nn.Linear(H2, H2)
        self.lin_bn4 = nn.BatchNorm1d(num_features=H2)
        self.linear5 = nn.Linear(H2, H)
        self.lin_bn5 = nn.BatchNorm1d(num_features=H)
        self.linear6 = nn.Linear(H, D_in)
        self.lin_bn6 = nn.BatchNorm1d(num_features=D_in)

        self.relu = nn.ReLU()

    def encode(self, x):

        lin1 = self.relu(self.lin_bn1(self.linear1(x)))
        lin2 = self.relu(self.lin_bn2(self.linear2(lin1)))
        lin3 = self.relu(self.lin_bn3(self.linear3(lin2)))

        fc1 = F.relu(self.fc1(lin3))

        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)

        return r1, r2

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_() # multiply log variance with 0.5, then in-place exponent --> gives std
            # eps is [dim, zdim] with all elements drawn from mean 0 and std 1
            eps = Variable(std.data.new(std.size()).normal_()) # std.data is tensor wrapped by std

            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        fc3 = self.relu(self.fc3(z))
        fc4 = self.relu(self.fc4(fc3))

        lin4 = self.relu(self.lin_bn4(self.linear4(fc4)))
        lin5 = self.relu(self.lin_bn5(self.linear5(lin4)))

        return self.lin_bn6(self.linear6(lin5))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        return self.decode(z), mu, logvar

class customLoss(nn.Module):
    def __init__(self):
        super(customLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")
    
    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self.mse_loss(x_recon, x)
        loss_KLD = -0.5* torch.sum(1+ logvar - mu.pow(2) - logvar.exp())

        return loss_MSE + loss_KLD

def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every linear layer in a model
    if classname.find('Linear') != -1:
        # get the number of the inputs 
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)

D_in = data_set.x.shape[1]
H=100
H2=50
model = Autoencoder(D_in, H, H2).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

loss_custom = customLoss()

# Train

epochs = 2000
log_interval = 50
val_losses = []
train_losses = []

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_custom(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    if epoch % 200 == 0:
        print('Epoch {}, Average Loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))
        
        train_losses.append(train_loss / len(train_loader.dataset))


for epoch in range(1, epochs + 1):
    train(epoch)

standardizer = train_loader.dataset.standardizer

model.eval()
test_loss = 0
with torch.no_grad():
    for i, data in enumerate(train_loader):
        data = data.to(device)
        recon_batch, mu, logvar = model(data)

# Get Embeddings 
mu_output = []
logvar_output = []

with torch.no_grad():
    for i, (data) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)

        mu_tensor = mu
        mu_output.append(mu_tensor)
        mu_result = torch.cat(mu_output, dim=0)

        logvar_tensor = logvar
        logvar_output.append(logvar_tensor)
        logvar_result = torch.cat(logvar_output, dim=0)

# Plot Embeddings 
print(mu_result)
features = mu_result.detach().cpu().numpy()

with open('features.pkl', 'wb') as f:
    np.save(f, features)





