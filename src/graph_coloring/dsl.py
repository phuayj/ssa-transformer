from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class GraphColorActionType(IntEnum):
    SELECT_NODE = 0
    ASSIGN_COLOR = 1
    PROPAGATE = 2
    BACKTRACK = 3
    DONE = 4


@dataclass(frozen=True)
class GraphColorAction:
    type: GraphColorActionType
    target: Optional[int] = None  # node index for SELECT, color for ASSIGN

    def __str__(self) -> str:
        if self.type == GraphColorActionType.SELECT_NODE:
            return f"SELECT_NODE({self.target})"
        if self.type == GraphColorActionType.ASSIGN_COLOR:
            return f"ASSIGN_COLOR({self.target})"
        return self.type.name

    @classmethod
    def select_node(cls, node: int) -> "GraphColorAction":
        if int(node) < 0:
            raise ValueError("node must be non-negative")
        return cls(type=GraphColorActionType.SELECT_NODE, target=int(node))

    @classmethod
    def assign_color(cls, color: int) -> "GraphColorAction":
        if int(color) < 1:
            raise ValueError("color must be >= 1")
        return cls(type=GraphColorActionType.ASSIGN_COLOR, target=int(color))

    @classmethod
    def propagate(cls) -> "GraphColorAction":
        return cls(type=GraphColorActionType.PROPAGATE)

    @classmethod
    def backtrack(cls) -> "GraphColorAction":
        return cls(type=GraphColorActionType.BACKTRACK)

    @classmethod
    def done(cls) -> "GraphColorAction":
        return cls(type=GraphColorActionType.DONE)

    def to_token(self) -> str:
        """Convert to string token for model input/output."""

        if self.type == GraphColorActionType.SELECT_NODE:
            if self.target is None:
                raise ValueError("SELECT_NODE missing target")
            return f"SELECT_NODE_{int(self.target)}"

        if self.type == GraphColorActionType.ASSIGN_COLOR:
            if self.target is None:
                raise ValueError("ASSIGN_COLOR missing target")
            return f"ASSIGN_COLOR_{int(self.target)}"

        return self.type.name

    @classmethod
    def from_token(cls, token: str) -> "GraphColorAction":
        """Parse from string token."""

        t = token.strip()
        if not t:
            raise ValueError("Empty action token")

        upper = t.upper()
        if upper in {"PROPAGATE", "BACKTRACK", "DONE"}:
            return cls(type=GraphColorActionType[upper])

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

        if upper.startswith("SELECT_NODE") or upper.startswith("SELECT"):
            prefix = "SELECT_NODE" if upper.startswith("SELECT_NODE") else "SELECT"
            node = _parse_int(t[len(prefix) :], what="SELECT_NODE")
            return cls.select_node(node)

        if upper.startswith("ASSIGN_COLOR") or upper.startswith("ASSIGN"):
            prefix = "ASSIGN_COLOR" if upper.startswith("ASSIGN_COLOR") else "ASSIGN"
            color = _parse_int(t[len(prefix) :], what="ASSIGN_COLOR")
            return cls.assign_color(color)

        raise ValueError(f"Unknown action token: {token!r}")


if __name__ == "__main__":
    # Smoke tests
    a = GraphColorAction.select_node(5)
    assert GraphColorAction.from_token(a.to_token()) == a

    b = GraphColorAction.assign_color(2)
    assert GraphColorAction.from_token(b.to_token()) == b

    for x in [GraphColorAction.propagate(), GraphColorAction.backtrack(), GraphColorAction.done()]:
        assert GraphColorAction.from_token(x.to_token()) == x

    print("dsl.py smoke test passed")
