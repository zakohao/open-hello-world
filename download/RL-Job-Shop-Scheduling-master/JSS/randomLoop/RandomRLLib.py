import numpy as np
import ray

from ray.rllib import Policy
from ray.rllib.agents import with_common_config
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.execution.rollout_ops import ParallelRollouts, SelectExperiences
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.utils import override
from ray.rllib.utils.typing import TrainerConfigDict
from ray.util.iter import LocalIterator

DEFAULT_CONFIG = with_common_config({})
哦

class RandomLegalPolicy(Policy):
    """Just pick a randomLoop legal action"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exploration = self._create_exploration()

    @override(Policy)
    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):

        obs_batch = restore_original_dimensions(np.array(obs_batch, dtype=np.float32), self.observation_space, tensorlib=np)

        def pick_legal_action(legal_action):
            return np.random.choice(len(legal_action), 1, p=(legal_action / legal_action.sum()))[0]
        return [pick_legal_action(x) for x in obs_batch['action_mask']], [], {}

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass


def execution_plan(workers: WorkerSet,
                   config: TrainerConfigDict) -> LocalIterator[dict]:
    rollouts = ParallelRollouts(workers, mode="async")

    # Collect batches for the trainable policies.
    rollouts = rollouts.for_each(
        SelectExperiences(workers.trainable_policies()))

    # Return training metrics.
    return StandardMetricsReporting(rollouts, workers, config)


RandomMaskedTrainer = build_trainer(
    name="RandomMasked",
    default_config=DEFAULT_CONFIG,
    default_policy=RandomLegalPolicy,
    execution_plan=execution_plan)
