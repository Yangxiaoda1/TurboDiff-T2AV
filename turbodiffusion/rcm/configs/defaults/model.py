# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from hydra.core.config_store import ConfigStore

from imaginaire.lazy_config import LazyCall as L
from rcm.models.t2a_model_distill_rcm import T2ADistillConfig_rCM, T2ADistillModel_rCM
from rcm.models.t2a_model_distill_sla import T2ADistillConfig_SLA, T2ADistillModel_SLA

try:
    from rcm.models.t2v_model_distill_rcm import T2VDistillConfig_rCM, T2VDistillModel_rCM
    from rcm.models.t2v_model_sla import T2VConfig_SLA, T2VModel_SLA
    _HAS_T2V = True
except Exception:
    _HAS_T2V = False

if _HAS_T2V:
    FSDP_CONFIG_T2V_DISTILL_RCM = dict(
        trainer=dict(distributed_parallelism="fsdp"),
        model=L(T2VDistillModel_rCM)(config=T2VDistillConfig_rCM(fsdp_shard_size=8), _recursive_=False),
    )

    FSDP_CONFIG_T2V_SLA = dict(
        trainer=dict(distributed_parallelism="fsdp"),
        model=L(T2VModel_SLA)(config=T2VConfig_SLA(fsdp_shard_size=8), _recursive_=False),
    )

DDP_CONFIG_T2A_DISTILL_SLA = dict(
    trainer=dict(distributed_parallelism="ddp"),
    model=L(T2ADistillModel_SLA)(config=T2ADistillConfig_SLA(), _recursive_=False),
)

DDP_CONFIG_T2A_DISTILL_RCM = dict(
    trainer=dict(distributed_parallelism="ddp"),
    model=L(T2ADistillModel_rCM)(config=T2ADistillConfig_rCM(), _recursive_=False),
)

def register_model():
    cs = ConfigStore.instance()
    if _HAS_T2V:
        cs.store(group="model", package="_global_", name="fsdp_t2v_distill_rcm", node=FSDP_CONFIG_T2V_DISTILL_RCM)
        cs.store(group="model", package="_global_", name="fsdp_t2v_sla", node=FSDP_CONFIG_T2V_SLA)
    cs.store(group="model", package="_global_", name="ddp_t2a_distill_sla", node=DDP_CONFIG_T2A_DISTILL_SLA)
    cs.store(group="model", package="_global_", name="ddp_t2a_distill_rcm", node=DDP_CONFIG_T2A_DISTILL_RCM)

    # IT2AV models depend on external MOVA code. Keep their registration optional so that
    # pure T2V/T2A workflows remain usable in environments without MOVA full-stack deps.
    try:
        from rcm.models.it2av_model_distill_rcm import IT2AVDistillConfig_rCM, IT2AVDistillModel_rCM
        ddp_it2av_distill_rcm = dict(
            trainer=dict(distributed_parallelism="ddp"),
            model=L(IT2AVDistillModel_rCM)(config=IT2AVDistillConfig_rCM(), _recursive_=False),
        )
        fsdp_it2av_distill_rcm = dict(
            trainer=dict(distributed_parallelism="fsdp"),
            model=L(IT2AVDistillModel_rCM)(config=IT2AVDistillConfig_rCM(), _recursive_=False),
        )
        cs.store(group="model", package="_global_", name="ddp_it2av_distill_rcm", node=ddp_it2av_distill_rcm)
        cs.store(group="model", package="_global_", name="fsdp_it2av_distill_rcm", node=fsdp_it2av_distill_rcm)
    except Exception:
        pass
