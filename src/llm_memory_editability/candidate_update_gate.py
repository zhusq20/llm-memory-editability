"""Condition the existing joint update gate on the full candidate H + gq."""

from .joint_update_gate import JointUpdateGateGPT


class CandidateUpdateGateGPT(JointUpdateGateGPT):
    """Change only the gate input; retain the old state and final raw-H readout."""

    def _update_boundary(self, previous, state):
        candidate = self._discoloop_boundary(state)
        if self.update_mode == "full":
            return candidate
        update = self.update_gate(candidate).to(state.dtype)
        return previous + update * (candidate - previous)
