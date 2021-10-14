from models import *

input_dim = 1
output_dim = 1

model = LinearRegressionModel(input_dim, output_dim)
criterion = nn.MSELoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
epochs = 100

X_train = np.array([i for i in range(11)], dtype=np.float32).reshape(-1, 1) # make a column vector
y_train = np.array([2*i + 1 for i in range(11)], dtype=np.float32).reshape(-1, 1) # make a column vector

for epoch in range(epochs):
    epoch += 1
    
    inputs = Variable(torch.from_numpy(X_train))
    labels = Variable(torch.from_numpy(y_train))
    
    optimizer.zero_grad()
    
    outputs = model(inputs)
    
    loss = criterion(outputs, labels)
    
    loss.backward()
    
    optimizer.step()
    
    print("epoch {}, loss {}".format(epoch, loss.data[0]))

predicted = model(Variable(torch.from_numpy(X_train))).data.numpy()

plt.clf()

plt.plot(X_train, y_train, 'go', label='True Data', alpha=0.5)
plt.plot(X_train, predicted, '--', label='prediction', alpha=0.5)

plt.legend(loc='best')
plt.show()

# torch.save(model.state_dict(), 'linear_regression_model.pkl')
# model.load_state_dict(torch.load('linear_regression_model.pkl'))
