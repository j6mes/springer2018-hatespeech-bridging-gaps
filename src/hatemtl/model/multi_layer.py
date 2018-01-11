from torch import nn


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()
        if type(hidden_dims) == int:
            print("Warning: you passed a single integer ({}) as argument "
                  "hidden_dims, but a list of integers specifying dimensions"
                  "for all hidden layers is expected. The model will now have"
                  "a single hidden layer with the dimensionality you "
                  "specified.".format(hidden_dims))
            hidden_dims = [hidden_dims]
        self.dimensionalities = [input_dim] + hidden_dims
        i = 0
        self.hidden = []
        for i in range(len(hidden_dims)):
            self.hidden.append(nn.Linear(self.dimensionalities[i],
                                         self.dimensionalities[i + 1]))
        # self.fc1 = nn.Linear(input_dim,hidden_dim)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(self.dimensionalities[i + 1], output_dim)

        for layer in self.hidden:
            nn.init.xavier_normal(layer.weight)
        nn.init.xavier_normal(self.fc2.weight)

    def forward(self, x):
        h = x
        for layer in self.hidden:
            h = self.dropout(self.tanh(layer(h)))
        return self.fc2(h)


class MTMLP(nn.Module):

    def __init__(self, input_dim, hidden_dims, output_dim_a, output_dim_b):
        super(MTMLP, self).__init__()
        if type(hidden_dims) == int:
            print("Warning: you passed a single integer ({}) as argument "
                  "hidden_dims, but a list of integers specifying dimensions"
                  "for all hidden layers is expected. The model will now have"
                  "a single hidden layer with the dimensionality you "
                  "specified.".format(hidden_dims))
            hidden_dims = [hidden_dims]
        self.dimensionalities = [input_dim] + hidden_dims
        i = 0
        self.hidden = []
        for i in range(len(hidden_dims)):
            self.hidden.append(nn.Linear(self.dimensionalities[i],
                                         self.dimensionalities[i + 1]))
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.2)
        self.fc2a = nn.Linear(self.dimensionalities[i + 1], output_dim_a)
        self.fc2b = nn.Linear(self.dimensionalities[i + 1], output_dim_b)

        for layer in self.hidden:
            nn.init.xavier_normal(layer.weight)
        nn.init.xavier_normal(self.fc2a.weight)
        nn.init.xavier_normal(self.fc2b.weight)

    def forward(self, x):
        h = x
        for layer in self.hidden:
            h = self.dropout(self.tanh(layer(h)))
        ret = (self.fc2a(h),self.fc2b(h))
        return ret
