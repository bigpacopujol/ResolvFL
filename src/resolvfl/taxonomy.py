from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Hashable, Iterable, List, Mapping, Sequence, Tuple

Node = Hashable

@dataclass(frozen=True)
class Taxonomy:
    """Finite rooted tree with deterministic child order."""
    root: Node
    children: Mapping[Node, Sequence[Node]]

    def __post_init__(self):
        parent: Dict[Node, Node] = {}
        nodes = {self.root}
        for u, ch in self.children.items():
            nodes.add(u)
            for v in ch:
                if v in parent:
                    raise ValueError(f"Node {v!r} has more than one parent")
                parent[v] = u
                nodes.add(v)
        if self.root in parent:
            raise ValueError("Root cannot have a parent")
        for n in nodes:
            if n != self.root and n not in parent:
                raise ValueError(f"Disconnected/non-root node without parent: {n!r}")
        object.__setattr__(self, "_parent", parent)
        object.__setattr__(self, "_nodes", tuple(nodes))

    @property
    def nodes(self) -> Tuple[Node, ...]:
        return self._nodes

    @property
    def internal_nodes(self) -> Tuple[Node, ...]:
        return tuple(n for n in self.nodes if len(self.children.get(n, ())) > 0)

    @property
    def leaves(self) -> Tuple[Node, ...]:
        return tuple(n for n in self.nodes if len(self.children.get(n, ())) == 0)

    def parent(self, node: Node) -> Node:
        return self._parent[node]

    def ancestors(self, node: Node) -> List[Node]:
        """Proper ancestors ordered root -> parent(node)."""
        out = []
        cur = node
        while cur != self.root:
            cur = self.parent(cur)
            out.append(cur)
        out.reverse()
        return out

    def path(self, node: Node) -> List[Node]:
        """Root-to-node path including both endpoints."""
        return self.ancestors(node) + [node]

    def descendants(self, node: Node, include_self: bool = False) -> List[Node]:
        out = [node] if include_self else []
        stack = list(self.children.get(node, ()))
        while stack:
            v = stack.pop()
            out.append(v)
            stack.extend(self.children.get(v, ()))
        return out

    def descendant_leaves(self, node: Node) -> List[Node]:
        return [v for v in self.descendants(node, include_self=True) if v in set(self.leaves)]

    def child_on_path(self, ancestor: Node, descendant: Node) -> Node:
        path = self.path(descendant)
        try:
            i = path.index(ancestor)
        except ValueError as exc:
            raise ValueError(f"{ancestor!r} is not an ancestor of {descendant!r}") from exc
        if i + 1 >= len(path):
            raise ValueError("ancestor and descendant must be distinct")
        return path[i + 1]


