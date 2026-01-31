"""Main application: framework + clients."""

from src.framework.api.app import create_app
from src.clients import (
    onboarding_router,
    admin_router,
    tasks_router,
    agents_router,
    batch_router,
    queue_router,
    prompts_router,
    graph_router,
    evaluation_router,
)

app = create_app(
    title="Gen AI Framework",
    description="Modular Gen AI API: API, LLM, RAG, agents, batch expense, queue, streaming, prompts, LangGraph",
    version="0.1.0",
)

# Mount domain clients
app.include_router(onboarding_router)
app.include_router(admin_router)
app.include_router(tasks_router)
app.include_router(agents_router)
app.include_router(batch_router)
app.include_router(queue_router)
app.include_router(prompts_router)
app.include_router(graph_router)
app.include_router(evaluation_router)
