import logging
from collections.abc import Callable
from typing import Any
from uuid import UUID

from mloda.core.abstract_plugins.components.credential_grant import CredentialGrant, grant_scope
from mloda.core.abstract_plugins.components.utils import failure_report
from mloda.core.abstract_plugins.hook_context import carried_gate_scope
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.components.credential_scrub import (
    current_registered_secrets,
    redact_exception,
    registered_secrets_scope,
)


logger = logging.getLogger(__name__)


def thread_worker(
    command: Any,
    cfw_register: Any,
    cfw: ComputeFramework,
    from_cfw: ComputeFramework | UUID | None,
    registered_secrets: frozenset[str] = frozenset(),
    identity_secrets: frozenset[str] | None = None,
    grant: CredentialGrant | None = None,
    mark_done: Callable[[UUID], None] | None = None,
    in_gate_scope: bool = False,
) -> None:
    with (
        registered_secrets_scope(registered_secrets, identity_secrets),
        grant_scope(grant),
        carried_gate_scope(in_gate_scope),
    ):
        try:
            command.execute(cfw_register, cfw, from_cfw=from_cfw)
            if mark_done is not None:
                mark_done(command.uuid)
        except BaseException as e:  # not only Exception: anything escaping a thread is lost and the run hangs
            redact_exception(e, current_registered_secrets())
            msg, exc_info = failure_report(e)
            cfw_register.set_error(msg, exc_info, exception=e)
