"""Gate the complete nonfinal handoff, including the paper re-encoding.

The actual incoming state X already contains the previous injection. Compute
the unchanged full Transformer H=F(X), then P=H+g(H)q(H), and hand off
X+u(H)(P-X). The final round still reads raw H. No historical model is changed.
"""

from .paper_update_gate import PaperUpdateGateGPT


class JointUpdateGateGPT(PaperUpdateGateGPT):
    """Same parameters and initialization as the continuous-only update gate."""

    def _update_boundary(self, previous, state):
        paper_state = self._discoloop_boundary(state)
        if self.update_mode == "full":
            return paper_state
        update = self.update_gate(state).to(state.dtype)
        return previous + update * (paper_state - previous)
