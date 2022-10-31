from .multiagentenv import MultiAgentEnv

from copy import deepcopy
import numpy as np
from gym.utils import seeding
from collections import defaultdict


class Coin1DEnv(MultiAgentEnv):

    def __init__(self,
                 seed=None,
                 obs_last_action=False,
                 obs_instead_of_state=False,
                 obs_timestep_number=False,
                 obs_local=True,
                 window=2,
                 state_last_action=True,
                 state_timestep_number=False,
                 reward_sparse=False,
                 reward_local=False,
                 reward_scale=True,
                 reward_scale_rate=1,
                 p=1.0,
                 episode_limit=50,
                 n_agents=2,
                 ):

        self.n_agents = n_agents
        self.episode_limit = episode_limit

        self.obs_last_action = obs_last_action
        self.obs_instead_of_state = obs_instead_of_state
        self.obs_timestep_number = obs_timestep_number
        self.obs_local = obs_local
        self.window = window
        self.state_last_action = state_last_action
        self.state_timestep_number = state_timestep_number

        self.reward_sparse = reward_sparse
        self.reward_local = reward_local
        self.reward_scale = reward_scale
        self.reward_scale_rate = reward_scale_rate
        self.p = p

        print("SEED:", seed)
        self._seed = seed
        self.seed()

        self.n_actions = 3

        self._episode_steps = 0
        self.last_action = np.zeros((self.n_agents, self.n_actions))
        self.states = []
        self.min_state, self.max_state = 0, 13
        self.coin_loc = self.np_random.randint(self.min_state, self.max_state)
        self.coin_counts = {i+1 : 0 for i in range(self.n_agents)}
        self.update_coin = False

    def step(self, actions):
        self._episode_steps += 1

        actions_int = [int(a) for a in actions]
        self.last_action = np.eye(self.n_actions)[np.array(actions_int)]

        directions = actions - 1
        self.states = np.clip(self.states + directions, a_min=self.min_state, a_max=self.max_state-1)

        reward = np.array([0.0] * self.n_agents)
        info = {"%d_counts" % (l + 1): 0.0 for l in range(self.n_agents)}
        if self.update_coin:
            self.coin_loc = self.np_random.randint(self.min_state, self.max_state)
            self.update_coin = False

        agents_at_goals = [l for l, s in enumerate(self.states) if s == self.coin_loc]
        if len(agents_at_goals):
            if not self.reward_sparse:
                for l in range(self.n_agents):
                    if len(agents_at_goals) == l+1:
                        for ag in agents_at_goals:
                            reward[ag] += self.p[l]
            self.coin_counts[len(agents_at_goals)] += 1
            self.update_coin = True
        if not self.reward_local:
            reward = reward.max()

        terminated = False
        if self._episode_steps >= self.episode_limit:
            terminated = True
            info["true_reward"] = sum([self.p[l] * self.coin_counts[l + 1] for l in
                                       range(self.n_agents)])
            info.update({"%d_counts" % (l + 1) : self.coin_counts[l + 1] for l in range(self.n_agents)})
            if self.reward_sparse:
                reward += sum([self.p[l]*self.coin_counts[l+1] for l in range(self.n_agents)])

        if self.reward_scale:
            reward *= self.reward_scale_rate
        return reward, terminated, info

    def get_obs(self):
        """ Returns all agent observations in a list """
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        unit = self.get_unit_by_id(agent_id, agent_id, is_ego=True)

        ally_feats_dim = self.get_obs_ally_feats_size()
        own_feats_dim = self.get_obs_own_feats_size()
        ally_feats = np.zeros(ally_feats_dim, dtype=np.float32)
        own_feats = np.zeros(own_feats_dim, dtype=np.float32)

        own_feats[:] = unit
        al_ids = [
            al_id for al_id in range(self.n_agents) if al_id != agent_id
        ]

        for i, al_id in enumerate(al_ids):
            al_unit = self.get_unit_by_id(al_id, agent_id, is_ego=False)
            ally_feats[i, :len(al_unit)] = np.array(al_unit)
            if self.obs_last_action:
                ally_feats[i, -1] = self.last_action[al_id]

        agent_obs = np.concatenate(
            (
                ally_feats.flatten(),
                own_feats.flatten(),
            )
        )
        if self.obs_timestep_number:
            agent_obs = np.append(agent_obs,
                                  self._episode_steps / self.episode_limit)
        return agent_obs

    def get_obs_own_feats_size(self):
        if self.obs_local:
            own_feats = 2*(2*self.window+1)
        else:
            own_feats = 2*(self.max_state - self.min_state)
        return own_feats

    def get_obs_ally_feats_size(self):
        if self.obs_local:
            nf_al = 2*self.window+1
        else:
            nf_al = 2*(self.max_state - self.min_state)
        if self.obs_last_action:
            nf_al += self.n_actions
        return self.n_agents - 1, nf_al

    def get_obs_size(self):
        own_feats = self.get_obs_own_feats_size()
        n_allies, n_ally_feats = self.get_obs_ally_feats_size()
        ally_feats = n_allies * n_ally_feats

        if self.obs_timestep_number:
            own_feats += 1
        return ally_feats + own_feats

    def get_state(self):
        if self.obs_instead_of_state:
            obs_concat = np.concatenate(self.get_obs(), axis=0).astype(
                np.float32
            )
            return obs_concat
        state = np.zeros((self.n_agents, 2*(self.max_state - self.min_state)))
        for al_id, al_unit in enumerate(self.states):
            state[al_id, :(self.max_state - self.min_state)] = np.eye(self.max_state - self.min_state)[int(al_unit)]
            state[al_id, (self.max_state - self.min_state):] = np.eye(self.max_state - self.min_state)[int(self.coin_loc)]
        state = state.flatten()
        if self.state_last_action:
            state = np.append(state, self.last_action.flatten())
        if self.state_timestep_number:
            state = np.append(state,
                              self._episode_steps)
        state = state.astype(dtype=np.float32)
        return state

    def get_state_size(self):
        """ Returns the shape of the state"""
        if self.obs_instead_of_state:
            return self.get_obs_size() * self.n_agents

        size = self.n_agents * 2 * (self.max_state - self.min_state)
        if self.state_last_action:
            size += self.n_agents * self.n_actions
        if self.state_timestep_number:
            size += 1
        return size

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return [1] * self.n_actions

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.n_actions

    def get_reward_size(self):
        return (self.n_agents,) if self.reward_local else (1,)

    def seed(self):
        """Returns the random seed used by the environment."""
        self.np_random, _ = seeding.np_random(self._seed)

        return self._seed

    def get_unit_by_id(self, a_id, r_id, is_ego=False):
        """Get unit by ID."""
        if not self.obs_local:
            agent = np.zeros(2*(self.max_state - self.min_state))
            agent[:(self.max_state - self.min_state)] = np.eye(self.max_state - self.min_state)[int(self.states[a_id])]
            agent[(self.max_state - self.min_state):] = np.eye(self.max_state - self.min_state)[int(self.coin_loc)]
        else:
            agent = np.zeros((2 if is_ego else 1, 2*self.window + 1))
            if abs(self.states[a_id] - self.states[r_id]) <= self.window:
                agent[0, int(self.states[a_id] - self.states[r_id] + self.window)] = 1.0
            if is_ego:
                # coin
                if abs(self.coin_loc - self.states[r_id]) <= self.window:
                    agent[1, int(self.coin_loc - self.states[r_id] + self.window)] = 1.0
                # walls
                if abs(self.min_state - self.states[r_id]) <= self.window:
                    agent[:, :int(self.min_state - self.states[r_id] + self.window)] = -1
                if abs((self.max_state - 1) - self.states[r_id]) <= self.window:
                    agent[:, int((self.max_state - 1) - self.states[r_id] + self.window):] = -1
        return agent.flatten()

    def reset(self):
        self.coin_loc = self.np_random.randint(self.min_state, self.max_state)
        self.coin_counts = {i+1 : 0 for i in range(self.n_agents)}
        self.update_coin = False

        self._episode_steps = 0
        self.last_action = np.zeros((self.n_agents, self.n_actions))
        self.states = self.np_random.randint(self.min_state, self.max_state, size=(self.n_agents,))
        return self.get_obs(), self.get_state()

    def get_stats(self):
        return {}
