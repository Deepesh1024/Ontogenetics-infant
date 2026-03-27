"""
Global Workspace — Baars' Global Workspace Theory Implementation

Phase 1: Passive message bus — all modules post and all outputs are broadcast.
Phase 5: Competition activated — only the highest-salience module broadcasts.

This is designed from Day 1 so every module uses the bus interface.
Competition logic is activated later without rewriting module interfaces.
"""

import numpy as np
from collections import OrderedDict


class MessageBus:
    """
    Central message bus implementing GWT.

    Phase 1 (passive): All module outputs are available to all consumers.
    Phase 5 (competitive): Only the winner's output is broadcast.
    """

    def __init__(self):
        self._modules = OrderedDict()       # module_name → latest output dict
        self._salience = OrderedDict()       # module_name → salience scalar
        self._competition_active = False     # Phase 1: False, Phase 5: True
        self._winner = None                  # current winning module name
        self._broadcast_vector = None        # concatenated broadcast representation
        self._history = []                   # timestamped competition results

    def register_module(self, name):
        """Register a module with the bus. Must be called before posting."""
        self._modules[name] = None
        self._salience[name] = 0.0

    def post(self, module_name, data, salience=0.0):
        """
        Post a module's output to the bus.

        Args:
            module_name: registered module name
            data: dict or numpy array — the module's output
            salience: scalar magnitude indicating urgency/importance
        """
        if module_name not in self._modules:
            raise KeyError(f"Module '{module_name}' not registered. Call register_module first.")
        self._modules[module_name] = data
        self._salience[module_name] = float(salience)

    def compete(self):
        """
        Run the competition. Called once per timestep after all modules post.

        Returns:
            winner_name: name of the winning module
            winner_data: the winning module's output
        """
        if not self._competition_active:
            # Phase 1: no competition, return None
            self._winner = None
            return None, None

        # Find module with highest salience
        best_name = None
        best_salience = -float("inf")
        for name, sal in self._salience.items():
            if self._modules[name] is not None and sal > best_salience:
                best_salience = sal
                best_name = name

        self._winner = best_name

        if best_name is not None:
            self._history.append({
                "winner": best_name,
                "salience": best_salience,
                "all_saliences": dict(self._salience),
            })

        return best_name, self._modules.get(best_name)

    def get_broadcast(self):
        """
        Get the current broadcast output.

        Phase 1 (passive): returns dict of ALL module outputs.
        Phase 5 (competitive): returns only the winner's output.
        """
        if not self._competition_active:
            # Phase 1: broadcast everything
            return {
                name: data
                for name, data in self._modules.items()
                if data is not None
            }
        else:
            # Phase 5: broadcast only winner
            if self._winner is not None and self._modules[self._winner] is not None:
                return {self._winner: self._modules[self._winner]}
            return {}

    def get_broadcast_vector(self):
        """
        Get a flat numpy vector of the broadcast for concatenation to module inputs.

        Phase 1: concatenates all module outputs (that are numpy arrays).
        Phase 5: returns only the winner's vector.
        """
        broadcast = self.get_broadcast()
        vectors = []
        for name, data in broadcast.items():
            if isinstance(data, np.ndarray):
                vectors.append(data.flatten())
            elif isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, np.ndarray):
                        vectors.append(v.flatten())

        if vectors:
            return np.concatenate(vectors)
        return np.array([], dtype=np.float32)

    def activate_competition(self):
        """Switch from Phase 1 (broadcast-all) to Phase 5 (winner-takes-all)."""
        self._competition_active = True

    def deactivate_competition(self):
        """Revert to Phase 1 broadcast mode."""
        self._competition_active = False

    @property
    def is_competitive(self):
        return self._competition_active

    @property
    def winner(self):
        return self._winner

    def get_module_names(self):
        return list(self._modules.keys())

    def get_competition_history(self, last_n=10):
        return self._history[-last_n:]

    def clear(self):
        """Clear all module outputs for a new timestep."""
        for name in self._modules:
            self._modules[name] = None
            self._salience[name] = 0.0
        self._winner = None
