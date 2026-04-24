"""Master test suite entrypoint aggregating the project test surface."""

from tests.step_tests.step_00_bootstrap_test import *  # noqa: F401,F403
from tests.step_tests.step_01_scaffold_test import *  # noqa: F401,F403
from tests.step_tests.step_02_models_test import *  # noqa: F401,F403
from tests.step_tests.step_03_incidents_test import *  # noqa: F401,F403
