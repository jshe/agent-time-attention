import torch as th
import torch.nn as nn
import torch.nn.functional as F

class CriticNet(nn.Module):
    def __init__(self, scheme, args, input_shape):
        super(CriticNet, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        self.input_shape = input_shape

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, args.critic_hidden_dim)
        self.fc2 = nn.Linear(args.critic_hidden_dim, args.critic_hidden_dim)
        self.fc3 = nn.Linear(args.critic_hidden_dim, 1)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        v = self.fc3(x)
        return v

class BasicDistributedCritic(nn.Module):
    def __init__(self, scheme, args):
        super(BasicDistributedCritic, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        input_shape = self._get_input_shape(scheme)
        self.critic = [CriticNet(scheme, args, input_shape) for i in range(self.n_agents)]

    def forward(self, batch, t=None):
        inputs = self._build_inputs(batch, t=t)
        res = [self.critic[i](inputs[i]) for i in range(self.n_agents)]
        v = th.stack(res, dim=2)
        return v

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []

        # observation
        inputs.append(batch["obs"][:, ts])

        inputs.append(th.eye(1, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, self.args.n_agents, -1))
        inputs = [th.cat([x[:, :, i].reshape(bs, max_t, -1) for x in inputs], dim=-1) for i in range(self.n_agents)]
        return inputs

    def _get_input_shape(self, scheme):
        # observation
        input_shape = scheme["obs"]["vshape"]
        # agent id
        input_shape += 1
        return input_shape

    def parameters(self):
        res = []
        for i in range(self.args.n_agents):
            res += list(self.critic[i].parameters())
        return res
