import math
import hashlib
import gym
from enum import IntEnum
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from .rendering import *

# Size in pixels of a tile in the full-scale human view
TILE_PIXELS = 32

# Map of color names to RGB values
COLORS = {
    'red'   : np.array([255, 0, 0]),
    'green' : np.array([0, 255, 0]),
    'blue'  : np.array([0, 0, 255]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0]),
    'grey'  : np.array([100, 100, 100]),
    'orange': np.array([255, 128, 0]),
    'cyan': np.array([0, 255, 255]),
    'magenta': np.array([255, 0, 255])
}

AGENT_COLORS = ['purple', 'yellow', 'orange', 'cyan', 'magenta']

COLOR_NAMES = sorted(list(COLORS.keys()))

# Used to map colors to integers
COLOR_TO_IDX = {
    'red'   : 0,
    'green' : 1,
    'blue'  : 2,
    'purple': 3,
    'yellow': 4,
    'grey'  : 5,
    'orange': 6,
    'cyan': 7,
    'magenta': 8
}

IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

# Map of object type to integers
OBJECT_TO_IDX = {
    'unseen'        : 0,
    'empty'         : 1,
    'wall'          : 2,
    'floor'         : 3,
    'door'          : 4,
    'key'           : 5,
    'ball'          : 6,
    'box'           : 7,
    'goal'          : 8,
    'lava'          : 9,
    'agent'         : 10,
}

IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

# Map of state names to integers
STATE_TO_IDX = {
    'open'  : 0,
    'closed': 1,
    'locked': 2,
}

# Map of agent direction indices to vectors
DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]

class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self, type, color):
        assert type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color
        self.type = type
        self.color = color
        self.contains = None

        # Initial position of the object
        self.init_pos = None

        # Current position of the object
        self.cur_pos = None

    def can_overlap(self):
        """Can the agent overlap with this?"""
        return False

    def can_pickup(self):
        """Can the agent pick this up?"""
        return False

    def can_contain(self):
        """Can this contain another object?"""
        return False

    def see_behind(self):
        """Can the agent see behind this object?"""
        return True

    def toggle(self, env, pos):
        """Method to trigger/toggle an action this object performs"""
        return False

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], 0)

    @staticmethod
    def decode(type_idx, color_idx, state):
        """Create an object from a 3-tuple state description"""

        obj_type = IDX_TO_OBJECT[type_idx]
        color = IDX_TO_COLOR[color_idx]

        if obj_type == 'empty' or obj_type == 'unseen':
            return None

        # State, 0: open, 1: closed, 2: locked
        is_open = state == 0
        is_locked = state == 2

        if obj_type == 'wall':
            v = Wall(color)
        elif obj_type == 'floor':
            v = Floor(color)
        elif obj_type == 'ball':
            v = Ball(color)
        elif obj_type == 'key':
            v = Key(color)
        elif obj_type == 'box':
            v = Box(color)
        elif obj_type == 'door':
            v = Door(color, is_open, is_locked)
        elif obj_type == 'goal':
            v = Goal()
        elif obj_type == 'lava':
            v = Lava()
        elif obj_type == 'agent':
            v = Agent(color)
        else:
            assert False, "unknown object type in decode '%s'" % obj_type

        return v

    def render(self, r):
        """Draw this object with the given renderer"""
        raise NotImplementedError

class Goal(WorldObj):
    def __init__(self):
        super().__init__('goal', 'green')

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

class Floor(WorldObj):
    """
    Colored floor tile the agent can walk over
    """

    def __init__(self, color='blue'):
        super().__init__('floor', color)

    def can_overlap(self):
        return True

    def render(self, img):
        # Give the floor a pale color
        color = COLORS[self.color] / 2
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), color)


class Lava(WorldObj):
    def __init__(self):
        super().__init__('lava', 'red')

    def can_overlap(self):
        return True

    def render(self, img):
        c = (255, 128, 0)

        # Background color
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)

        # Little waves
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0,0,0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0,0,0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0,0,0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0,0,0))

class Wall(WorldObj):
    def __init__(self, color='grey'):
        super().__init__('wall', color)

    def see_behind(self):
        return False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

class Agent(WorldObj):
    def __init__(self, color=None):
        super().__init__('agent', color)

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])

class Door(WorldObj):
    def __init__(self, color, is_open=False, is_locked=False):
        super().__init__('door', color)
        self.is_open = is_open
        self.is_locked = is_locked

    def can_overlap(self):
        """The agent can only walk over this cell when the door is open"""
        return self.is_open

    def see_behind(self):
        return self.is_open

    def toggle(self, env, pos):
        # If the player has the right key to open the door
        if self.is_locked:
            if np.any([isinstance(env.carryings[idx], Key) and env.carryings[idx].color == self.color for idx in range(env.n_agents)]):
                self.is_locked = False
                self.is_open = True
                return True
            return False

        self.is_open = not self.is_open
        return True

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""

        # State, 0: open, 1: closed, 2: locked
        if self.is_open:
            state = 0
        elif self.is_locked:
            state = 2
        elif not self.is_open:
            state = 1

        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], state)

    def render(self, img):
        c = COLORS[self.color]

        if self.is_open:
            fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0,0,0))
            return

        # Door frame and door
        if self.is_locked:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.06, 0.94, 0.06, 0.94), 0.45 * np.array(c))

            # Draw key slot
            fill_coords(img, point_in_rect(0.52, 0.75, 0.50, 0.56), c)
        else:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0,0,0))
            fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
            fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0,0,0))

            # Draw door handle
            fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c)

class Key(WorldObj):
    def __init__(self, color='blue'):
        super(Key, self).__init__('key', color)

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]

        # Vertical quad
        fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), c)

        # Teeth
        fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), c)

        # Ring
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0,0,0))

class Ball(WorldObj):
    def __init__(self, color='blue'):
        super(Ball, self).__init__('ball', color)

    def can_pickup(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])

class Box(WorldObj):
    def __init__(self, color, contains=None):
        super(Box, self).__init__('box', color)
        self.contains = contains

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]

        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0,0,0))

        # Horizontal slit
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)

    def toggle(self, env, pos):
        # Replace the box by its contents
        env.grid.set(*pos, self.contains)
        return True

class Grid:
    """
    Represent a grid and operations on it
    """

    # Static cache of pre-renderer tiles
    tile_cache = {}

    def __init__(self, width, height):
        assert width >= 3
        assert height >= 3

        self.width = width
        self.height = height

        self.grid = [None] * width * height

    def __contains__(self, key):
        if isinstance(key, WorldObj):
            for e in self.grid:
                if e is key:
                    return True
        elif isinstance(key, tuple):
            for e in self.grid:
                if e is None:
                    continue
                if (e.color, e.type) == key:
                    return True
                if key[0] is None and key[1] == e.type:
                    return True
        return False

    def __eq__(self, other):
        grid1  = self.encode()
        grid2 = other.encode()
        return np.array_equal(grid2, grid1)

    def __ne__(self, other):
        return not self == other

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def set(self, i, j, v):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        self.grid[j * self.width + i] = v

    def get(self, i, j):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        return self.grid[j * self.width + i]

    def horz_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.width - x
        for i in range(0, length):
            self.set(x + i, y, obj_type())

    def vert_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.height - y
        for j in range(0, length):
            self.set(x, y + j, obj_type())

    def wall_rect(self, x, y, w, h):
        self.horz_wall(x, y, w)
        self.horz_wall(x, y+h-1, w)
        self.vert_wall(x, y, h)
        self.vert_wall(x+w-1, y, h)

    def rotate_left(self):
        """
        Rotate the grid to the left (counter-clockwise)
        """

        grid = Grid(self.height, self.width)

        for i in range(self.width):
            for j in range(self.height):
                v = self.get(i, j)
                grid.set(j, grid.height - 1 - i, v)

        return grid

    def slice(self, topX, topY, width, height):
        """
        Get a subset of the grid
        """

        grid = Grid(width, height)

        for j in range(0, height):
            for i in range(0, width):
                x = topX + i
                y = topY + j

                if x >= 0 and x < self.width and \
                   y >= 0 and y < self.height:
                    v = self.get(x, y)
                else:
                    v = Wall()

                grid.set(i, j, v)

        return grid

    @classmethod
    def render_tile(
        cls,
        obj,
        agent_dirs=None,
        highlight=False,
        tile_size=TILE_PIXELS,
        subdivs=3
    ):
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache
        key = (tuple(agent_dirs), highlight, tile_size)
        key = obj.encode() + key if obj else key

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if obj != None:
            obj.render(img)

        # Overlay the agent on top
        for agent_idx, agent_dir in enumerate(agent_dirs):
            if agent_dir >= 0:
                tri_fn = point_in_triangle(
                    (0.12, 0.19),
                    (0.87, 0.50),
                    (0.12, 0.81),
                )

                # Rotate the agent based on its direction
                tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5*math.pi*agent_dir)
                fill_coords(img, tri_fn, COLORS[AGENT_COLORS[agent_idx]])

        # Highlight the cell if needed
        if highlight:
            highlight_img(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img

    def render(
        self,
        tile_size,
        agent_posits=None,
        agent_dirs=None,
        highlight_mask=None
    ):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(self.width, self.height), dtype=np.bool)

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)
                agent_heres = [np.array_equal(agent_pos, (i, j)) for agent_pos in agent_posits]
                tile_img = Grid.render_tile(
                    cell,
                    agent_dirs=[agent_dir if agent_here else -1 for agent_dir, agent_here in zip(agent_dirs, agent_heres)],
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size
                )

                ymin = j * tile_size
                ymax = (j+1) * tile_size
                xmin = i * tile_size
                xmax = (i+1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    def encode(self, vis_mask=None):
        """
        Produce a compact numpy encoding of the grid
        """

        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        array = np.zeros((self.width, self.height, 3), dtype='uint8')

        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    v = self.get(i, j)

                    if v is None:
                        array[i, j, 0] = OBJECT_TO_IDX['empty']
                        array[i, j, 1] = 0
                        array[i, j, 2] = 0

                    else:
                        array[i, j, :] = v.encode()

        return array

    @staticmethod
    def decode(array):
        """
        Decode an array grid encoding back into a grid
        """

        width, height, channels = array.shape
        assert channels == 3

        vis_mask = np.ones(shape=(width, height), dtype=np.bool)

        grid = Grid(width, height)
        for i in range(width):
            for j in range(height):
                type_idx, color_idx, state = array[i, j]
                v = WorldObj.decode(type_idx, color_idx, state)
                grid.set(i, j, v)
                vis_mask[i, j] = (type_idx != OBJECT_TO_IDX['unseen'])

        return grid, vis_mask

    def process_vis(grid, agent_pos):
        mask = np.zeros(shape=(grid.width, grid.height), dtype=np.bool)

        mask[agent_pos[0], agent_pos[1]] = True

        for j in reversed(range(0, grid.height)):
            for i in range(0, grid.width-1):
                if not mask[i, j]:
                    continue

                cell = grid.get(i, j)
                if cell and not cell.see_behind():
                    continue

                mask[i+1, j] = True
                if j > 0:
                    mask[i+1, j-1] = True
                    mask[i, j-1] = True

            for i in reversed(range(1, grid.width)):
                if not mask[i, j]:
                    continue

                cell = grid.get(i, j)
                if cell and not cell.see_behind():
                    continue

                mask[i-1, j] = True
                if j > 0:
                    mask[i-1, j-1] = True
                    mask[i, j-1] = True

        for j in range(0, grid.height):
            for i in range(0, grid.width):
                if not mask[i, j]:
                    grid.set(i, j, None)

        return mask


    def apply_vis(grid, mask):
        for j in range(0, grid.height):
            for i in range(0, grid.width):
                if not mask[i, j]:
                    grid.set(i, j, None)

        return mask


class MiniGridEnv(gym.Env):
    """
    2D grid world game environment
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 10
    }

    # Enumeration of possible actions
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2

        # Pick up an object
        pickup = 3
        # Drop an object
        drop = 4
        # Toggle/activate an object
        toggle = 5

        # Done completing task
        done = 6

    def __init__(
        self,
        grid_size=None,
        width=None,
        height=None,
        episode_limit=100,
        see_through_walls=False,
        seed=1337,
        agent_view_size=7,
        reward_sparse=None,
        reward_local=None,
        p=1.0,
        n_agents=2
    ):
        # Can't set both grid_size and width/height
        self.step_count = 0
        if grid_size:
            assert width == None and height == None
            width = grid_size
            height = grid_size

        # Action enumeration for this environment
        self.actions = MiniGridEnv.Actions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        # Number of cells (width and height) in the agent view
        assert agent_view_size % 2 == 1
        assert agent_view_size >= 3
        self.agent_view_size = agent_view_size

        # Observations are dictionaries containing an
        # encoding of the grid and a textual 'mission' string
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.agent_view_size, self.agent_view_size, 3),
            dtype='uint8'
        )
        self.observation_space = spaces.Dict({
            'image': self.observation_space
        })

        # Range of possible rewards
        self.reward_range = (0, 1)

        # Window to use for human rendering mode
        self.window = None

        # Environment configuration
        self.width = width
        self.height = height
        self.episode_limit = episode_limit
        self.see_through_walls = see_through_walls

        self.reward_sparse = reward_sparse
        self.reward_local = reward_local
        self.p = p#[float(n) for n in p.split(',')]
        self.n_agents = n_agents

        # Current position and direction of the agent
        self.agent_posits = [None]*self.n_agents
        self.agent_dirs = [None]*self.n_agents
        self.goal_counts = {i+1 : 0 for i in range(self.n_agents)}
        self.update_goals = []

        # Initialize the RNG
        print("SEED ", seed)
        self._seed = seed
        self.seed()

        # Initialize the state
        self.reset()

    def reset(self):
        # Current position and direction of the agent
        self.agent_posits = [None]*self.n_agents
        self.agent_dirs = [None]*self.n_agents
        self.goal_counts = {i+1 : 0 for i in range(self.n_agents)}
        self.update_goals = []


        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        assert np.all([agent_pos is not None for agent_pos in self.agent_posits])
        assert np.all([agent_dir is not None for agent_dir in self.agent_dirs])

        # Check that the agent doesn't overlap with an object
        for agent_pos in self.agent_posits:
            start_cell = self.grid.get(*agent_pos)
            assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.carryings = [None]*self.n_agents

        # Step count since episode start
        self.step_count = 0

        # # Return first observation
        # obs = [self.gen_obs(agent_idx) for agent_idx in range(self.n_agents)]
        # return obs
        return self.get_obs(), self.get_state()

    def seed(self):
        # Seed the random number generator
        self.np_random, _ = seeding.np_random(self._seed)
        return self._seed

    def hash(self, size=16):
        """Compute a hash that uniquely identifies the current state of the environment.
        :param size: Size of the hashing
        """
        sample_hash = hashlib.sha256()

        to_encode = [self.grid.encode()] + self.agent_posits + self.agent_dirs
        for item in to_encode:
            sample_hash.update(str(item).encode('utf8'))

        return sample_hash.hexdigest()[:size]

    @property
    def steps_remaining(self):
        return self.episode_limit - self.step_count

    def __str__(self):
        """
        Produce a pretty string of the environment's grid along with the agent.
        A grid cell is represented by 2-character string, the first one for
        the object and the second one for the color.
        """

        # Map of object types to short string
        OBJECT_TO_STR = {
            'wall'          : 'W',
            'floor'         : 'F',
            'door'          : 'D',
            'key'           : 'K',
            'ball'          : 'A',
            'box'           : 'B',
            'goal'          : 'G',
            'lava'          : 'V',
        }

        # Short string for opened door
        OPENDED_DOOR_IDS = '_'

        # Map agent's direction to short string
        AGENT_DIR_TO_STR = {
            -1: '-', # unknown
            0: '>',
            1: 'V',
            2: '<',
            3: '^'
        }

        str = ''
        for agent_pos, agent_dir in zip(self.agent_posits, self.agent_dirs):
            for j in range(self.grid.height):

                for i in range(self.grid.width):
                    if i == agent_pos[0] and j == agent_pos[1]:
                        str += 2 * AGENT_DIR_TO_STR[agent_dir]
                        continue

                    c = self.grid.get(i, j)

                    if c == None:
                        str += '  '
                        continue

                    if c.type == 'door':
                        if c.is_open:
                            str += '__'
                        elif c.is_locked:
                            str += 'L' + c.color[0].upper()
                        else:
                            str += 'D' + c.color[0].upper()
                        continue

                    str += OBJECT_TO_STR[c.type] + c.color[0].upper()

                if j < self.grid.height - 1:
                    str += '\n'

        return str

    def _gen_grid(self, width, height):
        assert False, "_gen_grid needs to be implemented by each environment"

    def _reward(self):
        """
        Compute the reward to be given upon success
        """
        return sum([self.p[l]*self.goal_counts[l+1] for l in range(self.n_agents)])

    def _rand_int(self, low, high):
        """
        Generate random integer in [low,high[
        """

        return self.np_random.randint(low, high)

    def _rand_float(self, low, high):
        """
        Generate random float in [low,high[
        """

        return self.np_random.uniform(low, high)

    def _rand_bool(self):
        """
        Generate random boolean value
        """

        return (self.np_random.randint(0, 2) == 0)

    def _rand_elem(self, iterable):
        """
        Pick a random element in a list
        """

        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]

    def _rand_subset(self, iterable, num_elems):
        """
        Sample a random subset of distinct elements of a list
        """

        lst = list(iterable)
        assert num_elems <= len(lst)

        out = []

        while len(out) < num_elems:
            elem = self._rand_elem(lst)
            lst.remove(elem)
            out.append(elem)

        return out

    def _rand_color(self):
        """
        Generate a random color name (string)
        """

        return self._rand_elem(COLOR_NAMES)

    def _rand_pos(self, xLow, xHigh, yLow, yHigh):
        """
        Generate a random (x,y) position tuple
        """

        return (
            self.np_random.randint(xLow, xHigh),
            self.np_random.randint(yLow, yHigh)
        )

    def place_obj(self,
        obj,
        top=None,
        size=None,
        reject_fn=None,
        max_tries=math.inf
    ):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError('rejection sampling failed in place_obj')

            num_tries += 1

            pos = np.array((
                self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                self._rand_int(top[1], min(top[1] + size[1], self.grid.height))
            ))

            # Don't place the object on top of another object
            if self.grid.get(*pos) != None:
                continue

            # Don't place the object where the agent is
            if np.any([np.array_equal(pos, agent_pos) for agent_pos in self.agent_posits]):
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(*pos, obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def put_obj(self, obj, i, j):
        """
        Put an object at a specific position in the grid
        """

        self.grid.set(i, j, obj)
        obj.init_pos = (i, j)
        obj.cur_pos = (i, j)

    def place_agent(
        self,
        agent_idx,
        top=None,
        size=None,
        rand_dir=True,
        max_tries=math.inf
    ):
        """
        Set the agent's starting point at an empty position in the grid
        """

        self.agent_posits[agent_idx] = None
        pos = self.place_obj(None, top, size, max_tries=max_tries)
        self.agent_posits[agent_idx] = pos

        if rand_dir:
            self.agent_dirs[agent_idx] = self._rand_int(0, 4)

        return pos

    # @property
    def dir_vec(self, agent_idx):
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """

        assert self.agent_dirs[agent_idx] >= 0 and self.agent_dirs[agent_idx] < 4
        return DIR_TO_VEC[self.agent_dirs[agent_idx]]

    # @property
    def right_vec(self, agent_idx):
        """
        Get the vector pointing to the right of the agent.
        """

        dx, dy = self.dir_vec(agent_idx)
        return np.array((-dy, dx))

    # @property
    def front_pos(self, agent_idx):
        """
        Get the position of the cell that is right in front of the agent
        """

        return self.agent_posits[agent_idx] + self.dir_vec(agent_idx)

    def get_view_coords(self, i, j, agent_idx):
        """
        Translate and rotate absolute grid coordinates (i, j) into the
        agent's partially observable view (sub-grid). Note that the resulting
        coordinates may be negative or outside of the agent's view size.
        """

        ax, ay = self.agent_posits(agent_idx)
        dx, dy = self.dir_vec(agent_idx)
        rx, ry = self.right_vec(agent_idx)

        # Compute the absolute coordinates of the top-left view corner
        sz = self.agent_view_size
        hs = self.agent_view_size // 2
        tx = ax + (dx * (sz-1)) - (rx * hs)
        ty = ay + (dy * (sz-1)) - (ry * hs)

        lx = i - tx
        ly = j - ty

        # Project the coordinates of the object relative to the top-left
        # corner onto the agent's own coordinate system
        vx = (rx*lx + ry*ly)
        vy = -(dx*lx + dy*ly)

        return vx, vy

    def get_view_exts(self, agent_idx):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        """

        # Facing right
        if self.agent_dirs[agent_idx] == 0:
            topX = self.agent_posits[agent_idx][0]
            topY = self.agent_posits[agent_idx][1] - self.agent_view_size // 2
        # Facing down
        elif self.agent_dirs[agent_idx] == 1:
            topX = self.agent_posits[agent_idx][0] - self.agent_view_size // 2
            topY = self.agent_posits[agent_idx][1]
        # Facing left
        elif self.agent_dirs[agent_idx] == 2:
            topX = self.agent_posits[agent_idx][0] - self.agent_view_size + 1
            topY = self.agent_posits[agent_idx][1] - self.agent_view_size // 2
        # Facing up
        elif self.agent_dirs[agent_idx] == 3:
            topX = self.agent_posits[agent_idx][0] - self.agent_view_size // 2
            topY = self.agent_posits[agent_idx][1] - self.agent_view_size + 1
        else:
            assert False, "invalid agent direction"

        botX = topX + self.agent_view_size
        botY = topY + self.agent_view_size

        return (topX, topY, botX, botY)

    def relative_coords(self, x, y, agent_idx):
        """
        Check if a grid position belongs to the agent's field of view, and returns the corresponding coordinates
        """

        vx, vy = self.get_view_coords(x, y, agent_idx)

        if vx < 0 or vy < 0 or vx >= self.agent_view_size or vy >= self.agent_view_size:
            return None

        return vx, vy

    def in_view(self, x, y, agent_idx):
        """
        check if a grid position is visible to the agent
        """

        return self.relative_coords(x, y, agent_idx) is not None

    def agent_sees(self, x, y, agent_idx):
        """
        Check if a non-empty grid position is visible to the agent
        """

        coordinates = self.relative_coords(x, y, agent_idx)
        if coordinates is None:
            return False
        vx, vy = coordinates

        obs = self.gen_obs(agent_idx)
        obs_grid, _ = Grid.decode(obs)#['image'])
        obs_cell = obs_grid.get(vx, vy)
        world_cell = self.grid.get(x, y)

        return obs_cell is not None and obs_cell.type == world_cell.type

    def step(self, actions):
        self.step_count += 1

        reward = np.array([0.0] * self.n_agents) if self.reward_local else 0.0
        info = {"%d_counts" % (l + 1): 0.0 for l in range(self.n_agents)}
        goals = []
        agents_at_goals = []
        fwd_posits = [self.front_pos(agent_idx) for agent_idx in range(self.n_agents)]
        if self.update_goals:
            for fwd_pos in list(set(self.update_goals)):
                self.grid.set(*np.asarray(fwd_pos), None)
                self._placeGoal()
            self.update_goals = []
        for agent_idx in range(self.n_agents):
            # Get the contents of the cell in front of the agent
            fwd_cell = self.grid.get(*fwd_posits[agent_idx])
            # Rotate left
            if actions[agent_idx] == self.actions.left:
                self.agent_dirs[agent_idx] -= 1
                if self.agent_dirs[agent_idx] < 0:
                    self.agent_dirs[agent_idx] += 4
            # Rotate right
            elif actions[agent_idx] == self.actions.right:
                self.agent_dirs[agent_idx] = (self.agent_dirs[agent_idx] + 1) % 4
            # Move forward
            elif actions[agent_idx] == self.actions.forward:
                if fwd_cell == None or fwd_cell.can_overlap():
                    self.agent_posits[agent_idx] = fwd_posits[agent_idx]
                if fwd_cell != None and fwd_cell.type == 'goal':
                    goals.append(tuple(fwd_posits[agent_idx]))
                    agents_at_goals.append(agent_idx)
                if fwd_cell != None and fwd_cell.type == 'lava':
                    pass

            # Pick up an object
            elif actions[agent_idx] == self.actions.pickup:
                if fwd_cell and fwd_cell.can_pickup():
                    if self.carryings[agent_idx] is None:
                        self.carryings[agent_idx] = fwd_cell
                        self.carryings[agent_idx].cur_pos = np.array([-1.0] * self.n_agents)
                        self.grid.set(*fwd_posits[agent_idx], None)

            # Drop an object
            elif actions[agent_idx] == self.actions.drop:
                if not fwd_cell and self.carryings[agent_idx]:
                    self.grid.set(*fwd_posits[agent_idx], self.carryings[agent_idx])
                    self.carryings[agent_idx].cur_pos = fwd_posits[agent_idx]
                    self.carryings[agent_idx] = None

            # Toggle/activate an object
            elif actions[agent_idx] == self.actions.toggle:
                if fwd_cell:
                    fwd_cell.toggle(self, fwd_posits[agent_idx])

        if goals:
            self.goal_counts[len(goals)] += 1
            self.update_goals = goals
            if not self.reward_sparse:
                for l in range(self.n_agents):
                    if len(goals) == l+1:
                        if self.reward_local:
                            for ag in agents_at_goals:
                                reward[ag] += self.p[l]
                        else:
                            reward += self.p[l]
        done = False
        if self.step_count >= self.episode_limit:
            done = True
            if self.reward_sparse:
                reward = self._reward()
            info.update({"%d_counts" % (l + 1) : self.goal_counts[l + 1] for l in range(self.n_agents)})

        return reward, done, info

    def gen_obs_grid(self, orig_agent_idx):
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        """
        topX, topY, botX, botY = self.get_view_exts(orig_agent_idx)
        grid = self.grid.slice(topX, topY, self.agent_view_size, self.agent_view_size)

        agent_grids = [Grid(self.grid.width, self.grid.height) for _ in range(self.n_agents)]
        for agent_idx in range(self.n_agents):
            if self.carryings[agent_idx]:
                agent_grids[agent_idx].set(*self.agent_posits[agent_idx], self.carryings[agent_idx])
            else:
                agent_grids[agent_idx].set(*self.agent_posits[agent_idx], Agent(color=AGENT_COLORS[agent_idx]))
        agent_grids = [agent_grid.slice(topX, topY, self.agent_view_size, self.agent_view_size) for agent_grid in agent_grids]

        for _ in range(self.agent_dirs[orig_agent_idx] + 1):
            grid = grid.rotate_left()
            agent_grids = [grid.rotate_left() for grid in agent_grids]

        # Process occluders and visibility
        # Note that this incurs some performance cost
        if not self.see_through_walls:
            vis_mask = grid.process_vis(agent_pos=(self.agent_view_size // 2, self.agent_view_size - 1))
        else:
            vis_mask = np.ones(shape=(grid.width, grid.height), dtype=np.bool)


        for grid in agent_grids:
            grid.apply_vis(vis_mask)
        agent_grids = [agent_grids[orig_agent_idx]] + agent_grids[:orig_agent_idx] + agent_grids[(orig_agent_idx+1):]
        return grid, vis_mask, agent_grids

    def gen_obs(self, agent_idx):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """

        grid, vis_mask, agent_grids = self.gen_obs_grid(agent_idx)

        # Encode the partially observable view into a numpy array
        image = np.reshape(grid.encode(vis_mask), [-1])
        agent_images = [np.reshape(agent_grid.encode(vis_mask), [-1]) for agent_grid in agent_grids]
        image = np.concatenate([image, *agent_images])

        assert hasattr(self, 'mission'), "environments must define a textual mission string"

        return image

    def get_obs_render(self, obs, tile_size=TILE_PIXELS//2):
        """
        Render an agent observation for visualization
        """

        grid, vis_mask = Grid.decode(obs)

        # Render the whole grid
        img = grid.render(
            tile_size,
            agent_pos=(self.agent_view_size // 2, self.agent_view_size - 1),
            agent_dir=3,
            highlight_mask=vis_mask
        )

        return img

    def render(self, mode='human', close=False, highlight=True, tile_size=TILE_PIXELS):
        """
        Render the whole-grid human view
        """
        return

    def close(self):
        if self.window:
            self.window.close()
        return

    def get_obs(self):
        """ Returns all agent observations in a list """
        obs = [self.gen_obs(agent_idx) for agent_idx in range(self.n_agents)]
        return obs

    def get_obs_size(self):
        """ Returns all agent observations in a list """
        return (self.agent_view_size**2)*(1+self.n_agents)*3


    def get_state_size(self):
        """ Returns all agent observations in a list """
        return (self.width*self.height+self.n_agents)*3

    def get_state(self):
        """ Returns all agent observations in a list """
        return np.concatenate([np.reshape(self.grid.encode(), [-1]), np.reshape(np.array(self.agent_posits), [-1]), np.array(self.agent_dirs)])

    def get_avail_actions(self):
        return [[1]*len(self.actions) for _ in range(self.n_agents)]

    def get_total_actions(self):
        return len(self.actions)

    def get_reward_size(self):
        return (self.n_agents,) if self.reward_local else (1,)

    def get_stats(self):
        return {}

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "reward_shape": self.get_reward_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

