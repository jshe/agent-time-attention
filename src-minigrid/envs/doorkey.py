from .minigrid import *
# from gym_minigrid.register import register

class DoorKeyEnv(MiniGridEnv):
    """
    Environment with a door and key, sparse reward
    """

    def __init__(self,
            size=8,
            episode_limit=0,
            seed=1337,
            agent_view_size=0,
             reward_sparse=None,
             reward_local=None,
             p=1.0,
             n_agents=2
                 ):
        super().__init__(
            grid_size=size,
            episode_limit=episode_limit,
            seed=seed,
            agent_view_size=agent_view_size,
            reward_sparse=reward_sparse,
            reward_local=reward_local,
            p=p,
            n_agents=n_agents,
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Create a vertical splitting wall
        splitIdx = self._rand_int(2, width-2)
        self.grid.vert_wall(splitIdx, 0)

        # Place a goal in the bottom-right corner
        self.place_obj(Goal(), top=(splitIdx+1, 0),
                       size=(width-splitIdx-1, height))

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        for agent_idx in range(self.n_agents):
            # Randomize the starting agent position and direction
            self.place_agent(agent_idx, size=(splitIdx, height))

        # Place a door in the wall
        doorIdx = self._rand_int(1, width-2)
        self.put_obj(Door('yellow', is_locked=True), splitIdx, doorIdx)

        # Place a yellow key on the left side
        self.place_obj(
            obj=Key('yellow'),
            top=(0, 0),
            size=(splitIdx, height)
        )
        self.splitIdx = splitIdx

        self.mission = "use the key to open the door and then get to the goal"

    def _placeGoal(self):
        splitIdx = self.splitIdx
        room_idx = self._rand_int(0, 2)
        if not room_idx:
            self.place_obj(Goal(), top=(0, 0),
                           size=(splitIdx, self.height))
        else:
            self.place_obj(Goal(), top=(splitIdx + 1, 0),
                           size=(self.width - splitIdx - 1, self.height))


class DoorKeyEnv5x5(DoorKeyEnv):
    def __init__(self,
                 size=5,
                 episode_limit=50,
                 seed=None,
                 window=2,
                 reward_sparse=False,
                 reward_local=False,
                 p=False,
                 n_agents=2,
                 ):
        super().__init__(
            size=size,
            episode_limit=episode_limit,
            seed=seed,
            agent_view_size=2 * window + 1,
            reward_sparse=reward_sparse,
            reward_local=reward_local,
            p=p,
            n_agents=n_agents,
        )

