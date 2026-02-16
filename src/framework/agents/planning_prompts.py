"""Centralized prompt templates for the planning agent."""

# -----------------------------------------------------------------------------
# Plan creation
# -----------------------------------------------------------------------------
PLAN_CREATE = """Create a detailed step-by-step plan to accomplish the following goal:

Goal: {goal}

Provide a numbered list of specific, actionable steps. Each step should be clear and executable.
Format: 
1. Step description
2. Step description
..."""

# -----------------------------------------------------------------------------
# Step execution
# -----------------------------------------------------------------------------
STEP_EXECUTE = """Execute this step: {step_description}

Context: Working towards goal: {goal}
Previous steps completed: {previous_steps}"""

# -----------------------------------------------------------------------------
# Plan revision (after failures)
# -----------------------------------------------------------------------------
PLAN_REVISE = """The following plan failed. Revise it to address the failures.

Original Goal: {original_goal}

Failed Steps:
{failed_steps}

Completed Steps:
{completed_steps}

Create a revised plan that addresses the failures and continues from where we left off."""

# -----------------------------------------------------------------------------
# Result compilation
# -----------------------------------------------------------------------------
RESULT_COMPILE = """Compile a final summary based on the completed plan execution.

Goal: {goal}

Completed Steps and Results:
{completed_steps_and_results}

Provide a comprehensive final answer."""


def format_plan_create(goal: str) -> str:
    return PLAN_CREATE.format(goal=goal)


def format_step_execute(
    step_description: str,
    goal: str,
    previous_steps: list,
) -> str:
    prev_str = str(previous_steps) if previous_steps else "None yet"
    return STEP_EXECUTE.format(
        step_description=step_description,
        goal=goal,
        previous_steps=prev_str,
    )


def format_plan_revise(
    original_goal: str,
    failed_steps_text: str,
    completed_steps_text: str,
) -> str:
    return PLAN_REVISE.format(
        original_goal=original_goal,
        failed_steps=failed_steps_text,
        completed_steps=completed_steps_text,
    )


def format_result_compile(goal: str, completed_steps_and_results: str) -> str:
    return RESULT_COMPILE.format(
        goal=goal,
        completed_steps_and_results=completed_steps_and_results,
    )
