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

