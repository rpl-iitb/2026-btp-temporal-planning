# New algorithm trials

Brainstorm (2026-11-15) for a more novel algorithm for the RSS submission: multi-robot temporal planning with
durative actions, deadlines and uncertain durations, on FactorySim and the kitchens. Each idea is a candidate to try.
Status (2026-11-16): section 3 is implemented as V14 (`algorithm/v14.py`) with ideas 2, 4, 10 (simplified) and 1
(partly); 3, 5-9 are not implemented.

## 1. Why a new algorithm

The current final pipeline (learned station selection, then a scheduler / RL policy over macro steps, V8/V9/V13,
plus replanning) is close to existing work:

| Prior work | What it already does |
| --- | --- |
| ScheduleNet (Wang & Gombolay, RSS 2020) | graph attention over a simple temporal network, imitation + Q-learning, multi-robot scheduling with temporal and spatial constraints, generalises from small to large problems |
| Dai & Gluzman (Stochastic Systems 2022) | PPO for queueing-network control, beats strong heuristics |
| replanning on deviation | standard practice |

What is ours and worth building on: the PDDL symbolic layer (resource detection, TRPG lower bounds, COMP_T) and
FactorySim.

## 2. Where time is in our graphs today

| Code | Graph | Time |
| --- | --- | --- |
| T-iFLAX (`algorithm/gnn.py`) | static object graph (PLOI/FLAX), facts as edges, encode-process-decode, sum aggregation, 3 steps, 16 hidden | 8 TRPG numbers per node |
| V13 (`algorithm/v13.py`) | none: MLP per candidate task + mean context | slack feature |
| FactorySim (`factory_sim/env.py` `graph()`) | stations, robots, jobs; edges `at`, `next`, `conveyor` | due time as a feature |

Time is a number on a node. It never decides which nodes talk to each other or how messages are combined
(before V14, `algorithm/v14.py`, where the graph is made of events and messages are max-plus).

## 3. Main idea: tropical event-graph networks

Scheduling time uses two operations: an event happens after the **latest** of its prerequisites (max) and an
activity ends a duration later (plus). This is the max-plus (tropical) algebra of timed event graphs (Baccelli et
al., *Synchronization and Linearity*); the PERT critical path is a max-plus path.

**Graph = events, not state.**
- Nodes: start / end of running activities, committed future tasks of each robot, candidate tasks, deadlines.
- Edges: temporal constraints. "Starts after that ends" (precedence, resource hand-off) weighted by the duration;
  "must finish within D" as a negative edge from the deadline.
- Built automatically from PDDL durative actions (they are already start / end pairs); robots' intentions are part
  of the graph.

```
 r1: [load j3 @MC1]──45s──▶[unload j3 @asm1]──20s──▶(r1 free)
                                │ resource hand-off
 MC1: (j3 done)──▶[start j5]──300s──▶[end j5]
 r2:  (free)·· candidate ··▶[pickup j5 @MC1] ──▶ ...      deadline(j5) ◀─(−D)─ [kit j5]
```

**Message passing = learned max-plus.**

    t_v ← max over edges (u → v) of ( t_u + learned_delay(u, v) )   + learned residual

- One layer contains one critical-path / Bellman-Ford step exactly; at initialisation the network is CPM.
- Algorithmic alignment (Xu et al.): a network shaped like the algorithm needs fewer samples and generalises to
  larger instances (train on S, test on L).
- Earliest times, slack, critical path and lateness come from the structure; a deadline violation is a positive
  cycle, so infeasibility can be detected, not only guessed.
- The same machinery for the kitchens and the factory, from PDDL.

**Related work to position against.**

| Work | Difference |
| --- | --- |
| Tropical Attention (arXiv 2505.17190) | max-plus transformers for general algorithmic reasoning, not scheduling or robots |
| Graph-RHO (arXiv 2604.10073) | job-shop GNN with a critical-path auxiliary loss, no max-plus layers |
| ScheduleNet (RSS 2020) | STN graph but ordinary attention, no uncertainty, not from PDDL |

The combination (event graph from PDDL + max-plus layers + uncertain durations + multiple robots) looks open.

## 4. Other ideas to try

**A. Graph building**
1. **Durative actions as hyperedges.** An activity joins a robot, a job, a machine and an operator for an interval.
   T-iFLAX splits arity-3 facts into pairwise edges and loses that. Use a bipartite graph of object nodes and
   activity nodes, built from the PDDL schemas.
2. **Candidate tasks as nodes.** A decision is "pick a node". Each candidate is inserted into the event graph and one
   or two max-plus rounds give its effect on everyone's slack: a learned what-if at the cost of one forward pass.
3. **Resource timelines as chains.** Each machine and robot gets an ordered chain of its reservations, so messages
   also run along a resource's future.

**B. Message passing**
4. **Max-plus layers with a learned residual.** Never worse than CPM at start; learning adds what CPM ignores
   (contention, travel, blocking).
5. **Backpressure as message passing.** Queue nodes send pressure (backlog + urgency) upstream; the network equals
   the max-weight rule at initialisation, so it keeps its stability argument, and training improves on it.
6. **Event-driven memory (Temporal Graph Networks, Rossi et al. 2020).** Update a node only when an event touches it
   (machine done, robot arrived). Cheap, matches discrete-event simulation and ROS event streams, and remembers a
   robot or machine that keeps running slower than nominal, so the policy adapts online (implicit calibration on
   hardware). Not used for robot scheduling as far as the search showed.

**C. Multiple agents**
7. **Intentions as structure.** Each robot's next 1–2 committed tasks are event nodes; when r2 decides it sees r1's
   plan as constraints, not as features.
8. **Sequential choice within a round.** Robots that decide at the same instant choose one after another, each choice
   inserted into the graph before the next; no hand-written "is another robot closer" feature.
9. **Delay imposed on others.** Score a choice by how much it pushes back other robots' events (marginal cost, VCG
   style), computed by message passing; a principled multi-agent credit signal.

**D. Uncertain durations**
10. **Distributions as messages.** Events carry completion-time distributions (mean and variance, or quantiles);
    "plus" adds means and variances, "max" uses Clark's approximation. Output: probability of meeting each
    deadline; decisions maximise it. A stochastic critical path computed by a GNN.

## 5. Queueing ideas (earlier brainstorm)

1. **PDDL → queueing network.** Servers = detected resources (machines, robots, operators, pots); job classes =
   goal-object flows; routes = action order in the relaxed plan; service times = durations. Control with queueing
   policies; POPF only fills in short action sequences. Nobody seems to compile temporal PDDL to a queueing network.
2. **Fluid plan + tracking.** LP over continuous flow (milliseconds): makespan lower bound and a target trajectory;
   robots track it. Asymptotically optimal for stochastic job shops (Bertsimas & Sethuraman; Nazarathy & Weiss).
   Re-solving under noise is almost free.
3. **Deadline-aware backpressure for moving servers.** Free robot picks the task with the largest queue imbalance +
   urgency; learn only a few bias weights. Robots move, so standard proofs need extension (cf. message-ferry robots).
4. **Bottleneck-first release (drum-buffer-rope / CONWIP)**, bottleneck found from the lower-bound parts.
5. **Risk-of-lateness priority** instead of slack (the stochastic version of the kitchen burn look-ahead).

## 6. Candidate paper story

> **Tropical event-graph networks.** From temporal PDDL, build an event graph of the team's current and intended
> activities; pass messages in a learned max-plus algebra over completion-time distributions; each robot picks the
> candidate event that maximises the team's on-time probability. The network starts as the exact critical-path rule
> and learns contention and uncertainty on top; event-driven memory adapts to real robot speeds.

Experiments it suggests: train on S, test on L; calibration of predicted deadline-hit probability; online
adaptation on TurtleBots; comparison against CP-SAT, dispatch rules, ScheduleNet-style attention, V13.

## 7. Risks

- Max aggregation learns poorly with near-ties (Mirjanić et al.); use a soft max with temperature.
- Clark's approximation degrades with many parallel branches.
- The graph changes at every event; update only affected nodes (idea 6).
- Queueing guarantees are long-run; on small batch instances CP-SAT may still win on makespan.

## 8. Open question

Deadlines become "must finish within" edges. Should the network **guarantee** it never picks a task that provably
makes a deadline impossible (exact max-plus layers + hard check), or learn to weigh deadlines? This decides the
architecture and how the paper reads.

## Sources

- ScheduleNet, RSS 2020: https://roboticsproceedings.org/rss16/p094.html
- Dai & Gluzman, Queueing Network Controls via Deep RL: https://arxiv.org/abs/2008.01644
- Processing Network Controls via Deep RL: https://arxiv.org/pdf/2205.02119
- Tropical Attention: https://arxiv.org/pdf/2505.17190
- Graph-RHO: https://arxiv.org/pdf/2604.10073
- Tropical geometry of PERT: https://arxiv.org/pdf/1202.6457
- Learnable Petri Net Neural Network (max-plus): https://www.mdpi.com/2504-4990/7/3/100
- Xu et al., What Can Neural Networks Reason About?: https://arxiv.org/pdf/1905.13211
- Dudzik & Veličković, GNNs are dynamic programmers: https://papers.neurips.cc/paper_files/paper/2022/file/8248b1ded388fcdbbd121bcdfea3068c-Paper-Conference.pdf
- Mirjanić, Pascanu & Veličković (max aggregation, ties): https://proceedings.mlr.press/v231/mirjanic24a/mirjanic24a.pdf
- Combinatorial optimization and reasoning with GNNs: https://arxiv.org/pdf/2102.09544
- Rossi et al., Temporal Graph Networks: https://arxiv.org/abs/2006.10637
- Provably expressive temporal graph networks: https://arxiv.org/pdf/2209.15059
- Bertsimas & Sethuraman, fluid relaxations for job shop: https://www.mit.edu/~dbertsim/papers/Stochastic%20Networks/From%20fluid%20relaxations%20to%20practical%20algorithms%20for%20job%20shop%20scheduling-%20the%20makespan%20objective.pdf
- Stochastic job shop, fluid policy: https://ideas.repec.org/a/spr/jcomop/v36y2018i1d10.1007_s10878-018-0294-6.html
- Nazarathy & Weiss, fluid approach: https://cris.haifa.ac.il/en/publications/a-fluid-approach-to-large-volume-job-shop-scheduling/
- Backpressure routing: https://en.wikipedia.org/wiki/Backpressure_routing
- Drift-plus-penalty: https://en.wikipedia.org/wiki/Drift_plus_penalty
- Fine-grained backpressure for message-ferrying robots: https://anrg.usc.edu/www/papers/camera_ready_fgbp.pdf
- Hertle & Nebel, KI 2018: https://gki.informatik.uni-freiburg.de/papers/hertle_et_al_ki2018.pdf
