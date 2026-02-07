"""Domain clients: onboarding, admin, tasks, agents, batch, queue, prompts, graph, web_automation."""

from .onboarding.router import router as onboarding_router
from .admin.router import router as admin_router
from .tasks.router import router as tasks_router
from .agents.router import router as agents_router
from .batch.router import router as batch_router
from .queue.router import router as queue_router
from .prompts.router import router as prompts_router
from .graph.router import router as graph_router
from .evaluation.router import router as evaluation_router
from .web_automation.router import router as web_automation_router

__all__ = [
    "onboarding_router",
    "admin_router",
    "tasks_router",
    "agents_router",
    "batch_router",
    "queue_router",
    "prompts_router",
    "graph_router",
    "evaluation_router",
    "web_automation_router",
]
