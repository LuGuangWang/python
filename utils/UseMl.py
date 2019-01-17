import numpy as np

def layer_sizes(x, y):
    print("x shape:" , x.shape)
    print("y shape:", y.shape)
    n_x = x.shape[0]
    n_y = y.shape[0]
    return (n_x,n_y)


def initialize_parameters(n_x, n_h, n_y):
    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1), dtype=np.float32)
    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1), dtype=np.float32)

    parameters = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

    return parameters

def sigmoid(x):
    return 1 / 1 + np.exp(-x)


def forward_propagation(x, parameters):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    z1 = np.dot(w1, x) + b1
    a1 = np.tanh(z1)

    z2 = np.dot(w2, z1) + b2
    a2 = sigmoid(z2)

    cache = {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}

    print('=========cache:',cache)

    return a2, cache


def compute_cost(a2, y):
    m = y.shape[1]

    # logprobs = np.multiply(np.log(a2),y) + np.multiply(np.log(1-a2),1-y)
    logprobs = np.multiply(np.log(a2), y)

    cost = -1 / m * np.sum(logprobs)

    cost = np.squeeze(cost)

    return cost


def backward_propagation(parameters, cache, x, y):
    m = x.shape[1]

    w1 = parameters['w1']
    w2 = parameters['w2']

    a1 = cache['a1']
    a2 = cache['a2']

    dz2 = a2 - y
    dw2 = 1 / m * np.dot(dz2, a1.T)
    db2 = 1 / m * np.sum(dz2, axis=1, keepdims=True)

    dz1 = np.dot(w2.T, dz2) * (1 - np.power(a1, 2))
    dw1 = 1 / m * np.dot(dz1, x.T)
    db1 = 1 / m * np.sum(dz1, axis=1, keepdims=True)

    grads = {'dw1': dw1, 'db1': db1, 'dw2': dw2, 'db2': db2}

    return grads


def update_parameters(parameters, grads, learning_rate=1.2):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    dw1 = grads['dw1']
    db1 = grads['db1']
    dw2 = grads['dw2']
    db2 = grads['db2']

    w1 = w1 - dw1 * learning_rate
    b1 = b1 - db1 * learning_rate
    w2 = w2 - dw2 * learning_rate
    b2 = b2 - db2 * learning_rate

    parameters = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

    return parameters


x = np.array([[4, 6], [3, 9]])
y = np.array([[0], [1]])

np.random.seed(3)
n_x, n_y = layer_sizes(x, y)
parameters = initialize_parameters(n_x, 4, n_y)

for i in range(0, 1):
    a2, cache = forward_propagation(x, parameters)

    cost = compute_cost(a2, y)

    grads = backward_propagation(parameters, cache, x, y)

    parameters = update_parameters(parameters, grads)

    if i % 10 == 0:
        print("Cost after iteration %i: %f" % (i, cost))