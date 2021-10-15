from models import *

# Get dataset from MNIST
train_dataset = datasets.MNIST(root="./data",
                              train=True,
                              transform=transforms.ToTensor(),
                              download=True)
test_dataset = datasets.MNIST(root='./data',
                              train=False,
                              transform=transforms.ToTensor())

input_dim = 28*28
output_dim = 10
batch_size = 100
n_iters = 3000
num_epochs = int(n_iters / (len(train_dataset) / batch_size))

# Use DataLoader to make dataset iterable
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Create logreg model
model = LogisticRegressionModel(input_dim, output_dim)
criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, 28*28))
        labels = Variable(labels)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        optimizer.step()
        
        iter += 1
        
        if iter % 500 == 0:
            correct = 0
            total = 0
            
            for images, labels in test_loader:
                images = Variable(images.view(-1, 28*28))
                
                outputs = model(images)
                
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                
                correct += (predicted == labels).sum()
                
            accuracy = 100 * correct / total
            
            print("Iteration {}, Loss: {}, Accuracy: {}".format(iter, loss.data, accuracy))
            
# torch.save(model.state_dict(), "logreg_model.pkl")
