from .coma_learner import COMALearner
from .pg_learner import PGLearner

REGISTRY = {}

REGISTRY["coma_learner"] = COMALearner
REGISTRY["pg_learner"] = PGLearner

