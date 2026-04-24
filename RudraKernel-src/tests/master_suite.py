"""Master test suite entrypoint aggregating the project test surface."""

from tests.step_tests.step_00_bootstrap_test import *  # noqa: F401,F403
from tests.step_tests.step_01_scaffold_test import *  # noqa: F401,F403
from tests.step_tests.step_02_models_test import *  # noqa: F401,F403
from tests.step_tests.step_03_incidents_test import *  # noqa: F401,F403
from tests.step_tests.step_04_minimal_env_test import *  # noqa: F401,F403
from tests.step_tests.step_05_npc_test import *  # noqa: F401,F403
from tests.step_tests.step_06_trust_test import *  # noqa: F401,F403
from tests.step_tests.step_07_pathogen_test import *  # noqa: F401,F403
from tests.step_tests.step_08_r4_hacking_test import *  # noqa: F401,F403
from tests.step_tests.step_09_curriculum_test import *  # noqa: F401,F403
from tests.step_tests.step_10_trust_poisoning_test import *  # noqa: F401,F403
from tests.step_tests.step_11_temporal_test import *  # noqa: F401,F403
from tests.step_tests.step_12_confidence_test import *  # noqa: F401,F403
