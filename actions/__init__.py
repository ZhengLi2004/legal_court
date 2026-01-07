"""A package defining the atomic actions that agents can perform.

This package contains modules that define the specific, reusable actions that
constitute an agent's capabilities. These actions are derived from the `metagpt`
Action class and encapsulate a single, well-defined task, often involving an
LLM call with a specific prompt.

The actions are categorized by the type of agent that uses them:
-   `controller_actions.py`: High-level, strategic actions used by the
    `ArgumentController` role for planning and decision-making.
-   `worker_actions.py`: Tactical, task-oriented actions used by the `Worker`
    roles for information retrieval and analysis.
"""
