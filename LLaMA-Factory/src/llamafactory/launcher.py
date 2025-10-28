# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from llamafactory.train.tuner import run_exp  # use absolute import
from typing import Optional
import sys

def launch(hparam_exp_name: Optional[str]=None):
    run_exp(hparam_exp_name=hparam_exp_name)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        hparam_exp_name = sys.argv.pop(-1)
    else:
        assert len(sys.argv) == 2, f"PLS FIX LAUNCH CMD {sys.argv=}"
        hparam_exp_name = None
    launch(hparam_exp_name=hparam_exp_name)
