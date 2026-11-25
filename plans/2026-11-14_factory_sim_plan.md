# FactorySim: plan for a smart-factory simulator (Gym + Gazebo + hardware)

A simulator of the IIT Bombay smart factory (Prof. Vibhor Pandhare's lab), in which several mobile robots
(TurtleBots) move product trays between kitting, CNC machines, the SMC line and assembly stations. It is built
in three stages that share one configuration and one core logic: a fast **Gym** simulator for learning and
planning research, a **Gazebo + ROS 2** twin with real navigation, and **hardware** runs on TurtleBots.

Values marked **[ask Vibhor]** are unknown; placeholders are used until the lab provides them.

---

## 1. Goals and non-goals

**Goals**
1. A faithful *task-level* model of the factory: products, routes, machines, buffers, operators, robots,
   processing and travel times, deadlines, errors.
2. Standard research interfaces: **Gymnasium** (one decision at a time) and **PettingZoo** (multi-robot),
   with action masks, vector and graph observations.
3. A **PDDL 2.1 bridge**: export the current state as a temporal planning problem, and execute a plan in
   the simulator. This connects T-iFLAX, POPF/OPTIC and our learned planners to the same environment.
4. **Fixed or random durations** (processing, travel, operators), machine breakdowns and quality errors,
   each switchable from the configuration.
5. **One core, three backends**: the same factory logic runs with sampled times (Gym), with Gazebo/ROS 2
   events, and with real robot events. Results in simulation then mean something on hardware.
6. A **benchmark suite** (scenario generator, splits, baselines, metrics) released as open source with the paper.

**Non-goals**
- Physics of machining or assembly; robot arms; detailed tray manipulation. A TurtleBot carries a tray;
  loading and unloading are modelled as a timed hand-off at the station (by an operator or a fixture).
- Low-level control and localisation: on Gazebo and hardware, Nav2 does navigation; the simulator plans
  at the task level.

---

## 2. The factory being modelled (from the ME7002 slides and Prof. Vibhor's email)

**Stations**

| Station | Real equipment | Role | Notes |
| --- | --- | --- | --- |
| Supply store | store | raw parts (3D-printed components) | source |
| Kitting | kitting station + operator (MediaPipe-validated picking) | builds a tray with the product's BOM | needs an operator |
| MC1, MC2 | Siemens CNC (MC1 3-axis milling, MC2 2-axis turning) | gear machining, products PR1 and PR3 | one tray at a time |
| MC3, MC4 | Fanuc CNC | gear machining, products PR2 and PR4 | one tray at a time |
| Assembly stage 1 | operator + YOLOv8 vision check | first assembly | needs an operator |
| MC5 to MC9 | SMC manufacturing setup | MC5, then MC6 or MC7, then MC8 or MC9, then MC9_OUT | **[ask Vibhor]** RFID only at "MC5 IN" and "MC9 OUT": is it one conveyor line? |
| Assembly stage 2 | operator | final assembly | needs an operator |
| Inspection | camera + OpenCV dashboard | accept / reject | reject leads to rework or scrap **[ask Vibhor]** |
| Store (finished goods) | store | sink | |

**Products and routes** (fixed per product; the products compete for shared machines)

| Product | Colour | Route |
| --- | --- | --- |
| PR1 | red | Kitting → MC1 → MC2 → Asm1 → MC5 → MC6 → MC8 → MC9_OUT → Asm2 |
| PR2 | green | Kitting → MC3 → MC4 → Asm1 → MC5 → MC6 → MC9 → MC9_OUT → Asm2 |
| PR3 | blue | Kitting → MC1 → MC2 → Asm1 → MC5 → MC7 → MC8 → MC9_OUT → Asm2 |
| PR4 | white | Kitting → MC3 → MC4 → Asm1 → MC5 → MC7 → MC9 → MC9_OUT → Asm2 |

Contention: MC1/MC2 are shared by PR1 and PR3, MC3/MC4 by PR2 and PR4, Asm1, MC5 and Asm2 by all four.
Bill of materials per product: gears G1–G5, lower and upper case, 5 small pins, 1 big pin, handle, blade, in
the product's colour.

**Decisions a planner makes here**
- which order to kit next (sequencing and release);
- which robot carries which tray, and when (assignment and timing);
- where trays wait when a machine is busy (buffers);
- what to do after a breakdown or a quality reject (re-routing, rework);
- *(optional, configurable)* alternative machines if the lab allows them (e.g. MC6 or MC7 for any product).

**Constraints**: finish all orders within a time limit (or per-order deadlines); avoid errors (wrong tray at a
station, wrong sequence; the RFID workflow engine rejects these); one tray per machine and per robot; buffer
capacities; operators available at kitting and assembly.

---

## 3. Architecture

```
                  factory.yaml  (layout, stations, products, routes, time models, robots, orders)
                        |
                 +--------------+
                 |  factory_sim  |  core: state, rules, event queue, validity checks, RFID event log
                 |    (core)     |  -- pure Python, no Gym / ROS dependency
                 +------+-------+
        time source:    |  sampled (Gym)  |  ROS 2 events (Gazebo)  |  ROS 2 + real stations (hardware)
          +-------------+-----------+--------------+------------------+-------------------+
          |                         |                                 |                   |
   Gymnasium / PettingZoo     PDDL 2.1 bridge                   Gazebo backend      Hardware backend
   (training, evaluation)     (export state, run plans)         (TurtleBot3 + Nav2) (TurtleBot3 + Nav2,
                                                                                     operators, timers / MQTT)
          |                         |
   policies: rules, CP-SAT,   POPF, OPTIC, T-iFLAX,
   RL, V9/V13-style GNN       VAL-checked
```

**Key design decision: one core, pluggable time source.** The core never decides *how long* something takes.
A backend tells it when an activity ends: the Gym backend samples durations; the Gazebo backend reports
"robot r reached dock d" from Nav2; the hardware backend reports real arrivals and operator confirmations.
The rules, validity checks, decision points and logs are identical everywhere, so a policy trained in Gym
runs unchanged in Gazebo and on hardware.

**Engine**: a small discrete-event engine (priority queue of events) written for this project, not SimPy.
Reasons: decision points ("robot r is free, choose its next task") are explicit, stepping to the next decision
is simple, there are no generator-based processes to unwind when a backend reports events, and it is fast.
Our existing multi-agent PDDL engine (`tiflax/menv.py`) already proves this pattern.

---

## 4. Core model (state, events, rules)

**State**
- **Orders / jobs**: id, product, route, current step, location (station, buffer, robot), status (waiting,
  in process, in transit, done, rejected), release time, deadline.
- **Stations**: type, capacity, input and output buffers (capacity per station), state (idle, busy,
  blocked, down), current job, operator requirement.
- **Robots**: node, state (idle, moving, docking, loading, unloading, charging), carried tray, battery
  *(optional)*.
- **Operators**: pool per area (kitting, assembly), busy or free.
- **Floor graph**: waypoints and edges with lengths; one docking spot per station side; node / edge capacity
  for congestion (optional reservation, like the kitchen's floor tiles).

**Events**: order released; kit done; robot arrived; load / unload done; processing started / done;
breakdown / repair; inspection pass / fail; deadline passed.

**Rules (checked by the core, as the factory's RFID workflow engine does)**: a tray may enter only the next
station of its route; a machine processes one tray at a time; a robot carries one tray; buffers have
capacity; an operator must be present at kitting and assembly. Invalid actions are masked, never silently fixed.

**Time models (per activity, from the config)**: fixed; uniform; truncated normal; lognormal (mean, CV);
empirical (sampled from recorded logs). Travel time = shortest-path length ÷ speed × noise + docking time
(+ congestion wait). Breakdowns: exponential time between failures and repair time (MTBF / MTTR).
Quality: reject probability per station; reject → rework at the same station or scrap and re-kit.
**Deterministic mode** turns all noise off (nominal values), so plans can be VAL-checked.

---

## 5. Interfaces

**Two action levels**
- **Task level (default)**: when a robot is free it chooses a task: *transport tray j from its current place
  to its next station*, *go to station s*, *fetch a new kit*, *wait*, *charge*. Also a "release order k to
  kitting" decision. Masks list what is legal now. (This is the level at which V9/V13 learned well.)
- **Primitive level**: move to node, dock, load, unload, wait; matches PDDL actions one-to-one.

**Gymnasium**: `FactoryEnv-v0`; decisions are serialised (one free robot at a time, in a fixed order);
`step(action)` advances to the next decision; `info["action_mask"]`.
**PettingZoo (AEC)**: one agent per robot; agents act when free.

**Observations**
- vector: per station (state, queue lengths, time to finish), per robot (node, carrying, busy time left),
  per job (step, location, slack to deadline), global (time, completed, late);
- **graph**: nodes = stations, robots, jobs; edges = route successor, location, travel time; for GNN policies;
- PDDL snapshot (text), for symbolic planners.

**Reward (configurable)**: −Δt per step (makespan), −tardiness, +completion, −error / reject, −idle robots.

**Metrics (logged per episode)**: makespan; on-time completion rate; total and max tardiness; throughput;
machine utilisation; robot utilisation and distance; queue lengths; number of errors / rejects; decision
latency; number of replans.

**PDDL 2.1 bridge**
- A factory domain: `move`, `dock`, `load`, `unload`, `kit`, `process` (no agent: runs by itself),
  `assemble` (needs an operator), with deadlines as timed literals.
- `env.to_pddl()` writes the current state (facts true now, running activities' remaining effects at their
  nominal ends, deadlines shifted to now), as `tiflax/snapshot.py` already does for the kitchen.
- `env.run_plan(plan)` executes a temporal plan time-triggered (or with replanning on deviation), and
  plans are checked with VAL in deterministic mode.

**Logging and rendering**
- **RFID-style event log** in the factory's own vocabulary ("KITTING OUT", "MC1 IN", "MC1 OUT", …), so
  simulated runs and real runs can be compared line by line.
- pygame top-down view: floor plan, stations coloured by state, robots, trays in product colours
  (red, green, blue, white), queues; a Gantt timeline per station and robot; GIF / MP4 export; headless mode.

---

## 6. Scenarios and benchmark suite

**Generator** (seeded): number of robots, order mix and count, release pattern (batch / staggered / online
arrivals), deadline tightness, noise level, breakdown rate, reject rate, buffer sizes, operator count.

| Tier | Robots | Orders | Extras |
| --- | --- | --- | --- |
| S | 1 | 2–4 | fixed times |
| M | 2 | 6–10 | random times |
| L | 3–4 | 12–20 | random times, deadlines |
| Stress | 3–4 | 12–20 | breakdowns, rejects, online arrivals, tight deadlines |

Splits: train / validation / test, with test tiers larger than training (generalisation). Each instance gets a
reference schedule from a strong rule (as the kitchen's certificates) to set deadlines and normalise makespan.

**Baselines shipped with the simulator**
- dispatching rules: FIFO, EDD (earliest deadline), SPT (shortest processing time), nearest free robot;
- CP-SAT schedule with nominal times, executed open-loop and with replanning;
- POPF / OPTIC through the PDDL bridge (with T-iFLAX object selection);
- RL from scratch (masked PPO);
- our learned task-choice policy (V13-style: imitation + RL, GNN over robots × tasks).

---

## 7. Gazebo + ROS 2 layer

- **World generation** from `factory.yaml`: floor, walls, station blocks with labels and ArUco markers at the
  docking spots, so that the same marker-based localisation used in CS7001 works.
- **Robots**: 3–4 TurtleBot3 in separate namespaces, each with its own Nav2 stack and a shared map
  (from SLAM in Gazebo, then of the real lab).
- **Task executor node** per robot: a ROS 2 action server that executes one task (navigate to the dock,
  align on the marker, wait for load / unload) and reports completion.
- **Factory node**: runs the *same core* with the ROS time source: robot arrivals from the executors;
  machine processing simulated by timers sampled from the config (later: real machine / RFID events over MQTT).
- **Policy node**: the planner or learned policy, receiving the same observations as in Gym.
- **Calibration**: logs of travel and docking times from Gazebo (and later hardware) are fitted to
  distributions and written back into the config, so the Gym layer uses realistic times.
- **Sim-to-sim check**: the same scenario run in Gym (calibrated) and in Gazebo should give similar
  makespans and event orders; a large gap points to a modelling error.

---

## 8. Hardware deployment

1. Map the lab (SLAM), mark docking spots at the real (or mock) stations with ArUco markers.
2. Start with 1 robot, then 2, then 3–4. Machines first as **timers** (processing simulated; operator
   confirms load / unload on a tablet), later real machine signals via the lab's MQTT / RFID pipeline.
3. Run the same scenarios as in Gym / Gazebo; record the RFID-style logs.
4. Safety: speed limits, an emergency stop, human-occupied zones as blocked nodes.
5. Protocol: for each method, the same orders and seeds where applicable; 10+ runs per scenario; report
   on-time completion, makespan, replans, errors; video.

---

## 9. Software layout and quality

```
factory_sim/
  core/        engine.py (events), model.py (state), rules.py (validity), timing.py (time models)
  config/      factory_iitb.yaml (the lab), schema.json, scenarios/
  envs/        gym_env.py, pettingzoo_env.py, observations.py (vector, graph)
  pddl/        domain.pddl, export.py (state -> problem), execute.py (plan -> sim), val.py
  baselines/   rules.py, cpsat.py, popf.py, ppo.py
  render/      pygame_view.py, gantt.py, gif.py
  logs/        rfid_log.py, kpis.py
  ros/         factory_node.py, executor_node.py, policy_node.py, world_gen.py, launch/
  tests/
```

**Tests**: route correctness (no tray skips or repeats a station); conservation (trays never vanish or
duplicate); capacity rules; determinism with a fixed seed; deterministic-mode plans pass VAL; PDDL export →
POPF → execute → same final state; Gym and ROS time sources give identical event orders on a scripted run.

**Release**: `pip install factory-sim`; MIT licence; docs with a 5-minute quickstart; example notebooks;
the IIT Bombay layout and a few generic layouts.

---

## 10. Milestones

Today: mid-November 2026. RSS deadlines are usually late January / early February, so about 11 weeks.

| Weeks | Milestone | Done when |
| --- | --- | --- |
| 1 | **M0 Spec**: parameters with Prof. Vibhor; floor plan; config schema | `factory_iitb.yaml` filled (placeholders allowed) |
| 1–2 | **M1 Core**: engine, model, rules, time models, rule baselines, RFID log | tests pass; FIFO/EDD run all tiers |
| 3 | **M2 Interfaces**: Gymnasium, PettingZoo, masks, vector + graph observations, pygame view, GIFs | masked PPO trains on tier S |
| 4 | **M3 PDDL bridge**: domain, export, plan execution, VAL; POPF / T-iFLAX on tier S–M | exported plans VAL-valid and executable |
| 4–5 | **M4 Benchmark**: generator, splits, CP-SAT baseline, KPIs | baseline table on all tiers |
| 5–7 | **M5 Learning**: V13-style GNN task choice (imitation + RL), replanning loop; ablations | beats rules on L / Stress |
| 5–7 | **M6 Gazebo** (in parallel): world generation, multi-robot Nav2, executor + factory nodes | a full scenario runs in Gazebo; timings logged |
| 8 | **M7 Calibration**: fit times from Gazebo; sim-to-sim check | Gym and Gazebo makespans agree within ~10 % |
| 8–10 | **M8 Hardware**: 1 → 2 → 3–4 TurtleBots; scenarios; videos | 10+ runs per method per scenario |
| 10–11 | **M9 Paper + release** | paper submitted; repository public |

**Critical path**: M1 → M2 → M5 (the learning results) and M6 → M8 (hardware). Gazebo starts as soon as the
core's interface is fixed (end of week 2). **Fallback** if hardware slips: report Gym + Gazebo results with
calibrated times, hardware as a demonstration.

---

## 11. Questions for Prof. Vibhor

1. Processing times per machine and product (mean and spread); setup times when the product changes.
2. Is MC5–MC9 one line with its own conveyor (trays move inside it on their own), or do trays move between
   MC5…MC9 individually? Can PR1/PR3 use MC7, or PR2/PR4 use MC6 (alternative machines)?
3. Typical order mix and batch size; target completion time / deadlines.
4. Buffer space at each machine; how many trays exist.
5. Operators: how many at kitting and assembly; are they shared?
6. Error types and rates (wrong tray, wrong sequence, quality reject); what happens after a reject.
7. Machine breakdown / maintenance frequency and duration (his robust-scheduling work suggests data exists).
8. Floor plan with distances (or permission to measure); where robots could dock; human walkways.
9. Can we read machine and RFID events (MQTT / OPC UA / PostgreSQL) for calibration and, later, for live runs?
10. Which KPIs matter most to the lab: makespan, on-time rate, utilisation, errors?

---

## 12. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Parameters arrive late | placeholders from the slides; config makes swapping trivial |
| Real stations can't be integrated | machines as timers + operator confirmation; MQTT later |
| Docking accuracy on TurtleBots | ArUco-based alignment (already done in CS7001); generous dock zones |
| Sim-to-real gap in travel times | calibrate distributions from Gazebo, then from hardware logs |
| Multi-robot congestion / deadlock in narrow aisles | node / edge reservation in the core; Nav2 collision avoidance; one-way lanes if needed |
| Scope too large for the deadline | Gym + learning first (the paper's core); Gazebo in parallel; hardware as the last milestone with a fallback |
| Learned policy unsafe on hardware | the core masks invalid actions; speed limits and e-stop; supervised runs |

---

## 13. How this connects to the existing work

| Existing piece | Reused for |
| --- | --- |
| `tiflax/menv.py` (multi-agent stochastic engine, decision rounds, replanning) | design of the core's decision loop and replanning |
| `tiflax/snapshot.py` (state → PDDL) | the PDDL bridge |
| `overcooked_ai_v3/dispatcher.py` | template for rule baselines and the imitation teacher |
| `tiflax/v13.py` (task choice, imitation + RL) | the learned policy, retargeted to factory tasks |
| T-iFLAX (GNN object relevance, closure, temporal recovery) | object selection before POPF on the PDDL bridge |
| kitchen renderer / GIFs | the pygame view and Gantt timeline |
