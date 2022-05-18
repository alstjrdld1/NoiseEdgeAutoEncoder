import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 

from torchvision import datasets, transforms 
import torchvision.models as models 

import cv2
import numpy as np 
from random import sample 
import matplotlib.pyplot as plt 

class NoiseEdgeAutoEncoder(nn.Module):
  def __init__(self, input_dim, hidden_dim1, hidden_dim2):
    super(NoiseEdgeAutoEncoder, self).__init__()

    self.encoder = nn.Sequential(
        nn.Linear(input_dim, hidden_dim1),
        nn.ReLU(),
        nn.Linear(hidden_dim1, hidden_dim2),
        nn.ReLU()
    )
    self.noise = np.random.normal(0, 1, hidden_dim2)
    self.decoder = nn.Sequential(
        nn.Linear(hidden_dim2, hidden_dim1),
        nn.ReLU(),
        nn.Linear(hidden_dim1, input_dim),
        nn.ReLU()
    )

  def edgeDetect(self, x):
    edges = cv2.Laplacian(x, -1)
    return edges

  def forward(self, x):
    out = x.view(x.size(0), -1)
    edge = self.edgeDetect(x)
    edge = edge.view(x.size(0), -1)

    out = self.encoder(out)
    edge = self.encoder(edge)

    out = out + torch.Tensor(self.noise).cuda() + edge

    out = self.decoder(out)
    out = out.view(x.size())
    return out

  def get_codes(self, x):
    return self.encoder(x)

def train(model, Loss, optimizer, num_epochs, device):
  train_loss_arr = []
  test_loss_arr =[]

  best_test_loss = 999999999
  early_stop, early_stop_max = 0., 3.

  for epoch in range(num_epochs):
    epoch_loss = 0.

    for batch_X, _ in train_loader:
      batch_X = batch_X.to(device)
      optimizer.zero_grad()

      model.train()
      outputs = model(batch_X)
      train_loss = Loss(outputs, batch_X)
      epoch_loss += train_loss.data

      train_loss.backward()
      optimizer.step()

    train_loss_arr.append(epoch_loss / len(train_loader.dataset))

    if epoch % 10 == 0 :
      model.eval()
      torch.save(model.state_dict(), f'20220518{epoch}.pt')
      test_loss = 0.

      for batch_X, _ in test_loader:
        batch_X = batch_X.to(device)

        outputs = model(batch_X)
        batch_loss = Loss(outputs, batch_X)
        test_loss += batch_loss.data 

      test_loss = test_loss 
      test_loss_arr.append(test_loss)

      if best_test_loss > test_loss:
        beset_test_loss = test_loss
        early_stop = 0

        print('Epoch [{}/{}], Train LOsos : {:.4f}, Test Loss : {:.4f} *'. format(epoch, num_epochs, epoch_loss, test_loss))
      else:
        early_stop += 1
        print('Epoch [{}/{}], Train LOsos : {:.4f}, Test Loss : {:.4f} *'. format(epoch, num_epochs, epoch_loss, test_loss))
    if early_stop >= early_stop_max:
      break

if __name__ == "__main__":
    train_dataset = datasets.MNIST(root='./data', train=True, transform = transforms.ToTensor(), download = True)
    test_dataset = datasets.MNIST(root='./data' , train=False, transform = transforms.ToTensor())

    total_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

    normal_dataset = sample([total_dataset.__getitem__(idx) 
                            for idx in range(len(total_dataset))
                            if total_dataset.__getitem__(idx)[1] == 5], 6000)
    
    batch_size = 512
    num_epochs = 300
    learning_rate = 0.01

    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle=False)

    NEAE = NoiseEdgeAutoEncoder(28 * 28, 64, 32)
    
    loss = nn.MSELoss()
    optimizer = optim.Adam(NEAE.parameters(), lr = learning_rate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train(NEAE, Loss = loss, optimizer=optimizer, num_epochs=num_epochs, device=device)