# G-Memory (Legal Adapter) Architecture & Implementation Guide

This document outlines the architecture of the **Legal Adapter for G-Memory**, mapping the theoretical concepts from the **Adversarial Spiral Protocol (ASP)** methodology to their concrete Python implementations within the `mas/` package.

---

## 1. Core Data Structure: Shadow Graph ($\mathcal{G}_t$)

**Methodology**:
The **Shadow Graph** is a structured, semantic representation of the legal dispute. Unlike raw dialogue logs, it tracks the logical dependencies between:
*   **Facts ($F$)**: Evidence or events.
*   **Laws ($L$)**: Legal statutes.
*   **Claims ($C$)**: Arguments derived from Facts/Laws.
*   **Edges ($E$)**: Support ($\rightarrow$) or Conflict ($\dashv$) relations.

Each node has a status: $S \in \{\text{Hypothetical}, \text{Validated}, \text{Defeated}\}$.

**Implementation** (`mas/common.py`):
```python
@dataclass
class ShadowGraph:
    """
    Maintains the directed graph of the legal argument.
    Supports semantic deduplication and serialization.
    """
    graph: nx.DiGraph
    
    def add_node(self, content, node_type, agent_id, matcher=None):
        # Adds node with semantic checking (Eq 3: ShouldMerge)
        ...
```

---

## 2. Memory Architecture (G-Memory 3-Tier)

The system adapts the G-Memory three-tier hierarchy for legal reasoning.

### A. Interaction Graph (The Case History)
**Methodology**: Concrete historical cases used for **Associative Projection**.
**Implementation** (`mas/legal_memory.py`):
*   Stored as JSON metadata in **ChromaDB**.
*   Retrieved via `retrieve_memory` and deserialized into `ShadowGraph` objects.

### B. Query Graph (The Case Topology)
**Methodology**: A navigation layer to find semantically similar cases (K-Hop).
**Implementation** (`mas/task_layer.py`):
*   **In-Memory**: Uses `networkx` to store case connectivity (Topology).
*   **Storage**: Persisted as `case_graph.pkl`.
*   **Logic**: Updates topology based on ChromaDB ANN results.

### C. Insight Graph (Legal Strategies)
**Methodology**: Abstract legal strategies extracted from winning/losing patterns.
**Implementation** (`mas/insights_manager.py`):
*   **In-Memory**: Vector index of strategies.
*   **Storage**: `legal_insights.json`.
*   **Logic**: `extract_adversarial_insights` compares winning vs. losing subgraphs.

---

## 3. Workflow & Algorithms

### Algorithm 1: Execute ($\text{Execute}(a_t, \mathcal{G}_t)$)
**Methodology**: Agents output actions (ADD/LINK) to modify the graph.
**Implementation** (`mas/graph_ops.py`):
*   **Component**: `GraphExecutor`
*   **Logic**: Parses LLM output (Regex-based) into atomic graph operations.
    *   `ADD_FACT("...")` $\rightarrow$ `graph.add_node(...)`
    *   `LINK(A, B, SUPPORT)` $\rightarrow$ `graph.add_edge(...)`

### Algorithm 3: Associative Projection ($\mathcal{P}(\mathcal{G}_0, \mathcal{M})$)
**Methodology**: Initializing the graph by projecting relevant subgraphs from history.
**Implementation** (`mas/projection.py`):
*   **Component**: `GraphProjector`
*   **Logic**:
    1.  Match current facts with historical facts using `SemanticMatcher`.
    2.  Project connected **Laws** and **Claims** from history.
    3.  **Filter**: Only projects `SUPPORT` edges to minimize noise.

### Algorithm 4: Verdict Back-Propagation
**Methodology**: Updating the graph status based on the Judge's verdict.
**Implementation** (`mas/backprop.py`):
*   **Component**: `BackPropagator`
*   **Logic**:
    1.  Mark Winner's nodes as `VALIDATED`.
    2.  Mark Loser's nodes (if challenged) as `DEFEATED`.

---

## 4. System Interface (The Facade)

**Implementation** (`mas/legal_system.py`):
The `LegalSystem` class orchestrates all components, serving as the "Brain" for MetaGPT agents.

| Methodology Step | API Method | Description |
| :--- | :--- | :--- |
| **Initialize** | `new_case(context)` | Performs **Downward** (Projection) & **Upward** (Insight Retrieval) traversals. |
| **Action** | `execute_action(graph, agent, text)` | Parses agent text into graph updates. |
| **Judge** | `adjudicate(context, graph)` | Calls the Judge (LLM/Model) for a verdict. |
| **Learn** | `learn(context, graph, winner)` | Executes **BackProp**, stores case, and extracts new insights. |

---

## 5. Usage Example

```python
from mas.legal_system import LegalSystem

# 1. Initialize
sys = LegalSystem("./storage")

# 2. Start Case (Trigger Projection & Insight Retrieval)
context = "Defendant stole a wallet but claims it was borrowed."
graph, strategies = sys.new_case(context)

# 3. Agent Action (Plaintiff)
sys.execute_action(graph, "plaintiff", 'ADD_CLAIM("Intent to deprive")')

# 4. Agent Action (Defendant)
sys.execute_action(graph, "defendant", 'CHALLENGE(CLAIM_1, "Borrowing is not theft")')

# 5. Verdict & Learning
settled, winner = sys.adjudicate(context, graph)
if settled:
    sys.learn(context, graph, winner, "case_001")
```

---

## 6. Limitations & Methodology Alignment

#### **A. Saturation vs. Semantic Verdict (The $\Delta \Phi$ Approximation)**
*   **Methodology Theory**: The paper proposes terminating the adversarial spiral based on **Information Gain Saturation ($\Delta \Phi \approx 0$)**—a structural metric indicating that further debate yields no new valid graph nodes.
*   **Current Implementation**: The system currently relies on the **Judge's Semantic Verdict** (`LLMJudge.evaluate`) to determine termination.
*   **Deviation Note**: The theoretical "Information Saturation" is approximated by the LLM's subjective assessment of case clarity ("Is the evidence sufficient?"). The explicit calculation of graph entropy or node saturation rate is **not yet implemented** in the `Judge` module.

#### **B. The Missing Driver (Algorithm 2)**
*   **Methodology Theory**: **Algorithm 2 (ASP)** defines the turn-based orchestration: `Plaintiff` $\rightarrow$ `Defendant` $\rightarrow$ `Check Saturation`.
*   **Current Implementation**: The `mas` package serves as the **Memory & Reasoning Engine**, providing atomic capabilities (`execute`, `project`, `learn`). It **does not** contain the debate loop itself.
*   **Integration Plan**: The **ASP Loop (Algorithm 2)** will be implemented as a **MetaGPT Standard Operating Procedure (SOP)**, where MetaGPT Roles (`Plaintiff`, `Defendant`) call the `LegalSystem` API in turns. The `mas` package is the *Library*, MetaGPT is the *Application*.