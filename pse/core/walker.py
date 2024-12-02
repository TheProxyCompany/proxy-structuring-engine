import json
import logging
from typing import Any
from pse_core import Walker as CoreWalker

logger = logging.getLogger(__name__)


class Walker(CoreWalker):

    @property
    def current_value(self) -> Any:
        """Retrieve the accumulated walker value.

        Returns:
            The current value from transition or history, parsed into an appropriate type.
            Returns None if no value is accumulated.
        """
        return self._parse_value(self.raw_value) if self.raw_value else None

    def _parse_value(self, value: Any) -> Any:
        """Parse the given value into an appropriate Python data type.

        Args:
            value: The value to parse.

        Returns:
            The parsed value as a float, int, dict, list, or the original value.
        """
        if not isinstance(value, str):
            return value

        try:
            float_value = float(value)
            return int(value) if float_value.is_integer() else float_value
        except ValueError:
            pass

        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

        return value

    def __str__(self) -> str:
        return f"{self.acceptor}.Walker({self.transition_walker})" if self.transition_walker else self.__repr__()

    def _format_current_edge(self) -> str:
        target_state = f"--> ({'✅' if self.target_state == '$' else self.target_state})" if self.target_state else ""
        accumulated_value = self.raw_value or self.current_value
        return f"Current edge: ({self.current_state}) --{repr(accumulated_value) if accumulated_value else ''}{target_state}"

    def __repr__(self) -> str:
        """Return a structured string representation of the walker."""
        def _format_state_info() -> str:
            if self.current_state == 0:
                return ""
            state_info = f"State: {self.current_state}"
            return f"{state_info} ➔ {self.target_state if self.target_state != '$' else '✅'}" if self.target_state and self.current_state != self.target_state else state_info

        def _format_history_info() -> str:
            if not self.accepted_history:
                return ""
            history_values = [repr(w.current_value)[1:-1] for w in self.accepted_history if w.current_value]
            return f"History: {', '.join(history_values)}" if history_values else ""

        def _format_remaining_input() -> str:
            return f"Remaining input: {repr(self.remaining_input)[1:-1]}" if self.remaining_input else ""

        def _format_transition_info() -> str:
            if not self.transition_walker:
                return ""
            transition_repr = repr(self.transition_walker)
            return f"Transition: {transition_repr}" if "\n" not in transition_repr and len(transition_repr) < 40 else f"Transition:\n  {transition_repr.replace('\n', '\n  ')}"

        prefix = "✅ " if self.has_reached_accept_state() else ""
        suffix = " 🔄" if self._accepts_more_input else ""
        header = f"{prefix}{self.acceptor.__class__.__name__}.Walker{suffix}"

        info_parts = list(filter(None, [
            _format_state_info(),
            _format_history_info(),
            self._format_current_edge(),
            _format_remaining_input(),
            _format_transition_info(),
        ]))

        single_line = f"{header} ({', '.join(info_parts)})" if info_parts else f"{header}()"
        if len(single_line) <= 80:
            return single_line

        indent = "  "
        multiline_parts = [f"{header} {{"]
        for part in info_parts:
            multiline_parts.extend([indent + line for line in part.split("\n")])
        multiline_parts.append("}")
        return "\n".join(multiline_parts)
