import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from modules.redists.lstm import LSTMLayer
from modules.redists.transformer import TransformerLayer

def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape[:-1], -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot

class RUDDER(nn.Module):
    def __init__(self, scheme, buffer, args):
        super(RUDDER, self).__init__()
        self.args = args
        self.n_units = args.n_rr
        self.buffer = buffer
        self.return_scaling = args.return_scaling
        self.rr_batch_size = args.rr_batch_size
        self.continuous_pred_factor = args.continuous_pred_factor
        self.n_actions = args.n_actions

        if self.args.arch == 'lstm':
            self.seq_layer = LSTMLayer(args, in_features=(_get_input_shape(args, scheme) + args.n_actions)*args.n_agents, out_features=self.n_units,
                              w_ci=(lambda *args, **kwargs: nn.init.normal_(mean=0, std=0.1, *args, **kwargs), False),
                              w_ig=(False, lambda *args, **kwargs: nn.init.normal_(mean=0, std=0.1, *args, **kwargs)),
                              w_og=False,
                              b_ci=lambda *args, **kwargs: nn.init.normal_(mean=0, *args, **kwargs),
                              b_ig=lambda *args, **kwargs: nn.init.normal_(mean=0, *args, **kwargs),
                              b_og=False,
                              a_out=torch.tanh
                              )
        elif self.args.arch == 'transformer':
            self.seq_layer = TransformerLayer(args, in_features=(_get_input_shape(args, scheme) + args.n_actions)*args.n_agents, out_features=self.n_units)
        self.linear = nn.Linear(self.n_units, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=args.rr_lr, weight_decay=args.l2_regularization)

        print("NUM PARAMS", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, input, train=False):
        bs, t, _, _ = input.shape
        input = input.reshape(bs, t, -1)
        seq_out = self.seq_layer.forward(input, train=train)
        out = self.linear(seq_out)
        return out, out

    def redistribute_reward(self, batch, actions):
        # Prepare inputs
        obs = _build_inputs(self.args, batch)
        obs_var = Variable(torch.FloatTensor(obs)).detach()
        actions_var = Variable(torch.FloatTensor(actions.type(torch.FloatTensor))).detach()
        # Calculate predictions
        seq_out, _ = self.forward(torch.cat((obs_var[:, :-1], to_one_hot(actions_var, self.n_actions)), 3), train=False)
        pred_g0 = torch.cat([torch.zeros_like(seq_out[:, 0:1, :]), seq_out], dim=1)
        redistributed_reward = pred_g0[:, 1:] - pred_g0[:, :-1]
        # Scale reward back up as targets have been scaled.

        redistributed_reward = redistributed_reward * self.return_scaling
        return redistributed_reward.detach()

    # Trains the RR until -on average- the main loss is below 0.25.
    def train(self, episode):
        i = 0
        loss_average = 0.3
        while loss_average > self.args.rr_min_loss and (self.args.rr_max_step < 0 or i < self.args.rr_max_step):
            i += 1
            obs, actions, rewards, mask = self.buffer.sample(self.rr_batch_size)
            main_loss_np = self.train_step(obs, actions, rewards, mask)
            loss_average -= self.args.rr_step_size * (loss_average - main_loss_np)
            if main_loss_np > loss_average * 2:
                loss_average = main_loss_np
        return i

    def train_step(self, obs, actions, rewards, mask):
        self.optimizer.zero_grad()

        # Get samples from the lesson buffer and prepare them.
        obs_var = Variable(torch.FloatTensor(obs)).detach()
        actions_var = Variable(torch.FloatTensor(actions)).detach()
        rewards_var = Variable(torch.FloatTensor(rewards)).detach()
        mask_var = Variable(torch.FloatTensor(mask)).detach()


        # Scale the returns as they might have high / low values.
        returns = torch.sum(rewards_var, 1, keepdim=True) / self.return_scaling
        # Run the RR
        predicted_G0, _ = self.forward(torch.cat((obs_var[:, :-1], to_one_hot(actions_var, self.args.n_actions)), 3), train=True)

        # Loss calculations
        all_timestep_loss = (predicted_G0 * mask_var - returns.repeat(1, predicted_G0.size(1), predicted_G0.size(2)))**2

        # Loss at any position in the sequence
        aux_loss = self.continuous_pred_factor * ((all_timestep_loss*mask_var)/mask_var.sum(dim=1, keepdim=True)).sum()

        # RR is mainly trained on getting the final prediction of g0 right.
        main_loss = all_timestep_loss[range(self.rr_batch_size), mask[..., -1].sum(axis=1)[:] - 1].mean()

        # RR update and loss tracking
        rr_loss = main_loss + aux_loss
        rr_loss.backward()
        main_loss_np = main_loss.data.numpy()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.rr_gnorm)
        self.optimizer.step()
        return main_loss_np

def _build_inputs(args, batch):
    inputs = []
    inputs.append(batch["obs"][:])
    bs = inputs[0].shape[0]
    traj_length = inputs[0].shape[1]
    if args.obs_agent_id:
        inputs.append(torch.eye(args.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, traj_length, -1, -1))

    inputs = torch.cat(inputs, dim=-1)
    return inputs

def _get_input_shape(args, scheme):
    input_shape = scheme["obs"]["vshape"]
    if args.obs_agent_id:
        input_shape += args.n_agents
    return input_shape
