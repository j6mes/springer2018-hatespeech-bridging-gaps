from torch import nn

class MLP(nn.Module):

    def __init__(self,input_dim,hidden_dim,output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim,hidden_dim)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim,output_dim)

        nn.init.xavier_normal(self.fc1.weight)
        nn.init.xavier_normal(self.fc2.weight)

    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out




class MTMLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim_a, output_dim_b):
        super(MTMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim,hidden_dim)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.2)
        self.fc2a = nn.Linear(hidden_dim,output_dim_a)
        self.fc2b = nn.Linear(hidden_dim,output_dim_b)

        nn.init.xavier_normal(self.fc1.weight)
        nn.init.xavier_normal(self.fc2a.weight)
        nn.init.xavier_normal(self.fc2b.weight)

    def forward(self, x):
        h = self.fc1(x)
        h = self.tanh(h)
        h = self.dropout(h)
        ret = (self.fc2a(h),self.fc2b(h))
        return ret
