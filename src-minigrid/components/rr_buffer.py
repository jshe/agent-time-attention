import numpy as np
import torch as th

class RRBuffer:
    def __init__(self, scheme, size, max_time, args):
        self.args = args
        self.size = size
        # Samples, time, features
        self.obs_buffer = np.empty(shape=(size, max_time + 1, self.args.n_agents, _get_input_shape(args, scheme)))
        self.actions_buffer = np.empty(shape=(size, max_time, self.args.n_agents, 1))
        self.rewards_buffer = np.empty(shape=(size, max_time, 1))
        self.mask_buffer = np.empty(shape=(size, max_time, 1), dtype=np.int32)
        self.loss_buffer = np.empty(shape=(size,), dtype=np.float32)
        self.next_spot_to_add = 0
        self.buffer_is_full = False
        self.samples_since_last_training = 0

    # LSTM training does only make sense, if there are sequences in the buffer which have different returns.
    # LSTM could otherwise learn to ignore the input and just use the bias units.
    def different_returns_encountered(self):
        if self.buffer_is_full:
            return np.unique(self.rewards_buffer.sum(1)).shape[0] > 1
        else:
            return np.unique(self.rewards_buffer[:self.next_spot_to_add].sum(1)).shape[0] > 1

    # We only train if 64 samples are played by a random policy
    def full_enough(self):
        print("next spot to add!", self.next_spot_to_add)
        return self.buffer_is_full or self.next_spot_to_add > self.args.rr_start_size  # 1024

    # Add a new episode to the buffer
    def add(self, batch, actions, rewards, mask):
        obs = _build_inputs(self.args, batch)
        bs = obs.shape[0]
        traj_length = obs.shape[1]
        next_ind = self.next_spot_to_add
        self.next_spot_to_add = self.next_spot_to_add + bs
        if self.next_spot_to_add >= self.size:
            self.buffer_is_full = True
        self.next_spot_to_add = self.next_spot_to_add % self.size
        self.obs_buffer[next_ind:(next_ind + bs), :traj_length] = self._to_np(obs)
        self.obs_buffer[next_ind:(next_ind + bs), traj_length:] = 0
        self.actions_buffer[next_ind:(next_ind + bs), :traj_length - 1] = self._to_np(actions)
        self.actions_buffer[next_ind:(next_ind + bs), (traj_length -1):] = 0
        self.rewards_buffer[next_ind:(next_ind + bs), :traj_length - 1] = self._to_np(rewards)
        self.rewards_buffer[next_ind:(next_ind + bs), (traj_length -1):] = 0
        self.mask_buffer[next_ind:(next_ind + bs), :traj_length - 1] = self._to_np(mask)
        self.mask_buffer[next_ind:(next_ind + bs), (traj_length -1):] = 0

    # Choose <batch_size> samples uniformly at random and return them.
    def sample(self, batch_size):
        self.samples_since_last_training = 0
        if self.buffer_is_full:
            indices = np.random.choice(self.size, batch_size)
        else:
            indices = np.random.choice(self.next_spot_to_add, batch_size)
        return (self.obs_buffer[indices, :, :], self.actions_buffer[indices, :],
                self.rewards_buffer[indices, :], self.mask_buffer[indices, :]
)

    def can_redist(self):
        return self.different_returns_encountered() and self.full_enough()

    def _to_np(self, t):
        """Try to convert a tensor or numpy.ndarray t to a numpy.ndarray"""
        try:
            t = t.numpy()
        except TypeError:
            t = t.cpu().numpy()
        except AttributeError:
            t = np.asarray(t)
        except RuntimeError:
            t = t.clone().data.cpu().numpy()
        return t

def _build_inputs(args, batch):
    inputs = []
    inputs.append(batch["obs"][:])
    bs = inputs[0].shape[0]
    traj_length = inputs[0].shape[1]
    if args.obs_agent_id:
        inputs.append(th.eye(args.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, traj_length, -1, -1))
    inputs = th.cat(inputs, dim=-1)
    return inputs

def _get_input_shape(args, scheme):
    input_shape = scheme["obs"]["vshape"]
    if args.obs_agent_id:
        input_shape += args.n_agents
    return input_shape
