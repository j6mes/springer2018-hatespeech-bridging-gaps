from torch import nn

class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        self.idx = 0
        for module in args:
            self.append(module)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def append(self,module):
        self.add_module(str(self.idx), module)
        self.idx += 1

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)



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

        self.hidden = ListModule()
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
        self.hidden = ListModule()
        for i in range(len(hidden_dims)):
            self.hidden.append(nn.Linear(self.dimensionalities[i],
                                         self.dimensionalities[i + 1]))

            #self.register_parameter("hidden_{0}".format(i), self.hidden[-1])


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
