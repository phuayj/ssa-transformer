from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class SatActionType(IntEnum):
    SELECT_VAR = 0      # Choose next decision variable
    ASSIGN_VALUE = 1    # Choose True/False for selected var (target in {0,1})
    PROPAGATE = 2       # Run unit propagation to fixpoint
    BACKTRACK = 3       # Undo to previous decision level + record nogood
    DONE = 4            # Declare SAT (or UNSAT if conflict at level 0)


@dataclass(frozen=True)
class SatAction:
    type: SatActionType
    target: Optional[int] = None  # var index for SELECT_VAR, 0/1 for ASSIGN_VALUE

    def __str__(self) -> str:
        if self.type == SatActionType.SELECT_VAR:
            return f"SELECT_VAR({self.target})"
        if self.type == SatActionType.ASSIGN_VALUE:
            return f"ASSIGN_VALUE({self.target})"
        return self.type.name

    @classmethod
    def select_var(cls, var: int) -> "SatAction":
        if int(var) < 0:
            raise ValueError("var must be non-negative")
        return cls(type=SatActionType.SELECT_VAR, target=int(var))

    @classmethod
    def assign_value(cls, value: int) -> "SatAction":
        v = int(value)
        if v not in {0, 1}:
            raise ValueError("value must be 0/1 (0=False, 1=True)")
        return cls(type=SatActionType.ASSIGN_VALUE, target=v)

    @classmethod
    def propagate(cls) -> "SatAction":
        return cls(type=SatActionType.PROPAGATE)

    @classmethod
    def backtrack(cls) -> "SatAction":
        return cls(type=SatActionType.BACKTRACK)

    @classmethod
    def done(cls) -> "SatAction":
        return cls(type=SatActionType.DONE)

    def to_token(self) -> str:
        """Convert to string token for model input/output."""

        if self.type == SatActionType.SELECT_VAR:
            if self.target is None:
                raise ValueError("SELECT_VAR missing target")
            return f"SELECT_VAR_{int(self.target)}"

        if self.type == SatActionType.ASSIGN_VALUE:
            if self.target is None:
                raise ValueError("ASSIGN_VALUE missing target")
            return f"ASSIGN_VALUE_{int(self.target)}"

        return self.type.name

    @classmethod
    def from_token(cls, token: str) -> "SatAction":
        """Parse from string token."""

        t = token.strip()
        if not t:
            raise ValueError("Empty action token")

        upper = t.upper()
        if upper in {"PROPAGATE", "BACKTRACK", "DONE"}:
            return cls(type=SatActionType[upper])

        def _parse_int(rest: str, *, what: str) -> int:
            r = rest.strip()
            if r.startswith("_"):
                r = r[1:]
            if r.startswith("(") and r.endswith(")"):
                r = r[1:-1]
            r = r.strip()
            if r == "":
                raise ValueError(f"Invalid {what} token: {token!r}")
            try:
                return int(r)
            except ValueError as e:
                raise ValueError(f"Invalid {what} int in token: {token!r}") from e

        if upper.startswith("SELECT_VAR") or upper.startswith("SELECT"):
            prefix = "SELECT_VAR" if upper.startswith("SELECT_VAR") else "SELECT"
            var = _parse_int(t[len(prefix) :], what="SELECT_VAR")
            return cls.select_var(var)

        if upper.startswith("ASSIGN_VALUE") or upper.startswith("ASSIGN"):
            prefix = "ASSIGN_VALUE" if upper.startswith("ASSIGN_VALUE") else "ASSIGN"
            val = _parse_int(t[len(prefix) :], what="ASSIGN_VALUE")
            return cls.assign_value(val)

        raise ValueError(f"Unknown action token: {token!r}")


if __name__ == "__main__":
    # Smoke tests
    a = SatAction.select_var(7)
    assert SatAction.from_token(a.to_token()) == a

    b = SatAction.assign_value(1)
    assert SatAction.from_token(b.to_token()) == b

    for x in [SatAction.propagate(), SatAction.backtrack(), SatAction.done()]:
        assert SatAction.from_token(x.to_token()) == x

    print("dsl.py smoke test passed")
