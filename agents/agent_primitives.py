from typing import List, Dict, Any, Optional
from copy import deepcopy
import asyncio
import json
from utils.logger import logger

class AgentPrimitives:
    # ------------------------------------------------------------
    # agent context
    # ------------------------------------------------------------
    class AgentContext:
        def __init__(
            self,
            brief: Optional[str] = None,
            max_iterations: int = 5,
            **kwargs: Any
        ):
            self.brief = brief or ""
            self.max_iterations = max_iterations
            self.iteration = 0
            self.sources: List[str] = []
            self.notes: List[str] = []
            self.search_queries: List[str] = []
            self.pages: List[dict] = []
            self.max_out = 3

            # allow arbitrary attributes
            for k, v in kwargs.items():
                setattr(self, k, v)

    # ------------------------------------------------------------
    # node primitives
    # ------------------------------------------------------------
    class Node:
        def __init__(self, name: str):
            self.name = name

        async def run(self, ctx: "AgentPrimitives.AgentContext", inp: Any) -> Any:
            raise NotImplementedError

    class FuncNode(Node):
        def __init__(self, name: str, func):
            super().__init__(name)
            self.func = func

        async def run(self, ctx: "AgentPrimitives.AgentContext", inp: Any) -> Any:
            # gctx and ctx are now the same object
            try:
                return await self.func(ctx, inp)
            except TypeError:
                # fallback for funcs that don't take inp
                return await self.func(ctx)

    # ------------------------------------------------------------
    # graph executor
    # ------------------------------------------------------------
    class Graph:
        def __init__(self, shared_parallel: bool = True):
            self.shared_parallel = shared_parallel

        async def run(self, ctx: "AgentPrimitives.AgentContext", *steps):
            """
            Execute a sequence of steps in one pass.
            A step can be:
            - a Node instance
            - a node factory (callable returning a Node)
            - a list of Nodes/factories for parallel execution
            - nested lists are supported
            """

            def _serialize_context(context):
                d = {}
                for k, v in vars(context).items():
                    try:
                        d[k] = v if isinstance(v, (str, int, float, bool, type(None))) else str(v)
                    except Exception:
                        d[k] = f"<unserializable:{type(v).__name__}>"
                return json.dumps(d, indent=2)

            def _to_node(maybe):
                # Node instance
                if isinstance(maybe, AgentPrimitives.Node):
                    return maybe
                # callable node factory
                if callable(maybe):
                    created = maybe()
                    if not isinstance(created, AgentPrimitives.Node):
                        raise TypeError(f"Factory did not return a Node: {maybe}")
                    return created
                raise TypeError(f"Unsupported step type: {type(maybe)}")

            async def _run_step(step, last_output):
                # parallel group
                if isinstance(step, list):
                    # normalize each element to Node or nested list
                    items = []
                    for item in step:
                        if isinstance(item, list):
                            items.append(item)  # keep nested list for recursion
                        else:
                            items.append(_to_node(item))

                    # execute
                    if self.shared_parallel:
                        # run Nodes directly in parallel, recurse for nested lists
                        coros = []
                        for it in items:
                            if isinstance(it, list):
                                coros.append(_run_step(it, last_output))
                            else:
                                coros.append(it.run(ctx, last_output))
                        results = await asyncio.gather(*coros)
                    else:
                        # deep copy context for isolation across this group
                        from copy import deepcopy
                        coros = []
                        for it in items:
                            if isinstance(it, list):
                                coros.append(_run_step(it, last_output))
                            else:
                                coros.append(it.run(deepcopy(ctx), last_output))
                        results = await asyncio.gather(*coros)

                    # logger.info(f"[AgentGraph] Finished parallel group → context:\n{_serialize_context(ctx)}")
                    return results

                # single node or factory
                node = _to_node(step)
                out = await node.run(ctx, last_output)
                # logger.info(f"[AgentGraph] Finished node '{node.name}' → context:\n{_serialize_context(ctx)}")
                return out

            last_output = None
            for step in steps:
                last_output = await _run_step(step, last_output)

            return last_output


    @staticmethod
    def node_loop(nodes: List["AgentPrimitives.Node"], max_iterations: int = 5) -> "AgentPrimitives.Node":
        class LoopNode(AgentPrimitives.Node):
            def __init__(self, name: str, nodes: List["AgentPrimitives.Node"], max_iterations: int):
                super().__init__(name)
                self.nodes = nodes
                self.max_iterations = max_iterations

            async def run(self, ctx: "AgentPrimitives.AgentContext", inp: Any) -> Any:
                last_out = inp
                for i in range(self.max_iterations):
                    ctx.iteration = i

                    # Derive friendlier counters if present
                    section_count = len((getattr(ctx, "outline", {}) or {}).get("sections", []) or [])
                    section_ix = getattr(ctx, "iteration", 0)
                    review_iter = getattr(ctx, "_review_iter", None)
                    review_budget = getattr(ctx, "_review_budget", None)

                    for n in self.nodes:
                        last_out = await n.run(ctx, last_out)

                        # Build a human-readable suffix
                        suffix_parts = []
                        # For the section-oriented loops
                        if section_count:
                            # iteration is zero-based; display as 1-based for humans
                            suffix_parts.append(f"section {min(section_ix+1, section_count)}/{section_count}")
                        # For the editor review budget if present
                        if review_iter is not None and review_budget is not None:
                            suffix_parts.append(f"review pass {review_iter}/{review_budget}")
                        suffix = f" ({', '.join(suffix_parts)})" if suffix_parts else ""

                        logger.info(f"[AgentGraph] After node '{n.name}'{suffix}")

                        if isinstance(last_out, bool) and not last_out:
                            logger.info(f"[AgentGraph] Exiting loop early after node '{n.name}'.")
                            return last_out

                logger.warning("[AgentGraph] Reached max_iterations without break signal.")
                return last_out

        return LoopNode("loop", nodes, max_iterations)
