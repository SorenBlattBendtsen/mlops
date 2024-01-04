from models.model import MyAwesomeModel
import torch
import matplotlib.pyplot as plt


path = "./data/processed/"
train_images = torch.load(path + "train_images.pt")
train_labels = torch.load(path + "train_target.pt")

train = torch.utils.data.TensorDataset(train_images, train_labels)

# create dataloaders
train = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)

model = MyAwesomeModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.NLLLoss()
train_losses = []
model.train()
for e in range(10):
    print(f"Epoch {e+1}")
    for images, labels in train:
        images = images.view(images.shape[0], -1)
        optimizer.zero_grad()
        
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

#save plot of training loss
plt.plot(train_losses)
path = "./reports/figures/"
plt.savefig(path + 'train_losses.png')

#save model to subfolder in models folder
path = "./models/"
torch.save(model.state_dict(), path+'checkpoint.pth')
