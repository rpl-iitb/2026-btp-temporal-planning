"""Online bilevel / imperative-learning training for iFlax's object importance scorer
"""
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import pddlgym
from pddlgym.structs import LiteralConjunction

from guidance.gnn_guidance import GNNSearchGuidance
from gnn.gnn import setup_graph_net
from gnn.gnn_dataset import create_super_graph
from planning import PlanningFailure, PlanningTimeout

from fd_tracked import TrackedFD
from iflax_planner import IFlaxPlanner


class IFlaxGNNGuidance(GNNSearchGuidance):

    def __init__(self, num_train_problems, num_epochs, bce_pos_weight,
                 complementary_rules, relaxation_rules,
                 gamma=0.9, expansion_frac=0.6,
                 load_from_file=True, save_model_prefix=None,
                 log_file=None, seed=0):
        super().__init__(
            training_planner=None,  # unused: we plan inside the loop, not offline
            num_train_problems=num_train_problems,
            num_epochs=num_epochs,
            criterion_name="bce",
            bce_pos_weight=bce_pos_weight,
            load_from_file=load_from_file,
            load_dataset_from_file=False,
            dataset_file_prefix="unused",
            save_model_prefix=save_model_prefix or "/tmp/iflax_model",
            is_strips_domain=True,
        )
        self._complementary_rules = complementary_rules
        self._relaxation_rules = relaxation_rules
        self._gamma = gamma
        self._expansion_frac = expansion_frac
        self._log_file = log_file
        self._seed = seed

    def _setup_feature_indices(self, states):
        """Same vocabulary bookkeeping as
        GNNSearchGuidance._create_graph_dataset, minus building offline
        target graphs (targets are generated online instead)."""
        self._unary_types = set()
        self._unary_predicates = set()
        self._binary_predicates = set()
        for state in states:
            self._unary_types.update({o.var_type for o in state.objects})
            for lit in set(state.literals) | set(state.goal.literals):
                arity = lit.predicate.arity
                if arity == 1:
                    self._unary_predicates.add(lit.predicate)
                elif arity == 2:
                    self._binary_predicates.add(lit.predicate)
        self._unary_types = sorted(self._unary_types)
        self._unary_predicates = sorted(self._unary_predicates)
        self._binary_predicates = sorted(self._binary_predicates)

        G = self.wrap_goal_literal
        R = self.reverse_binary_literal

        self._node_feature_to_index = {}
        index = 0
        for t in self._unary_types:
            self._node_feature_to_index[t] = index
            index += 1
        for p in self._unary_predicates:
            self._node_feature_to_index[p] = index
            index += 1
        for p in self._unary_predicates:
            self._node_feature_to_index[G(p)] = index
            index += 1

        self._edge_feature_to_index = {}
        index = 0
        for p in self._binary_predicates:
            self._edge_feature_to_index[p] = index
            index += 1
        for p in self._binary_predicates:
            self._edge_feature_to_index[R(p)] = index
            index += 1
        for p in self._binary_predicates:
            self._edge_feature_to_index[G(p)] = index
            index += 1
        for p in self._binary_predicates:
            self._edge_feature_to_index[G(R(p))] = index
            index += 1

        self._num_node_features = len(self._node_feature_to_index)
        self._num_edge_features = len(self._edge_feature_to_index)

    def _forward(self, state):
        """One differentiable forward pass. Returns (logits[n_node, 1],
        node_to_objects)."""
        graph_input, node_to_objects = self._state_to_graph(state)
        batch = create_super_graph([graph_input])
        outputs = self._model(batch.copy())
        logits = outputs[-1]["nodes"]
        return logits, node_to_objects

    def _refresh_score_cache(self, state):
        """Runs one no-grad forward pass and populates the score cache that
        score_object() reads. Done once per task *before* planning starts,
        so the (possibly parallel) 3R recovery threads only ever read this
        cache instead of racing to trigger their own forward passes."""
        with torch.no_grad():
            logits, node_to_objects = self._forward(state)
            probs = torch.sigmoid(logits).numpy()
        self._last_object_scores = {o: float(probs[n][0]) for n, o in node_to_objects.items()}
        self._last_processed_state = state

    def train(self, train_env_name, timeout=20, mode="auto"):
        """mode:
          "train" -- always run the bilevel loop fresh (ignores any cached
              checkpoint) and save the result.
          "test"  -- never run the bilevel loop; requires a cached
              checkpoint to already exist and raises if it doesn't. Still
              rebuilds the feature vocabulary and model architecture from
              the training states (cheap and deterministic -- no planning
              involved), which is required for the checkpoint's weights to
              load into a same-shaped model.
          "auto"  -- original behavior: load the cached checkpoint if
              load_from_file=True and one exists, else train fresh.
        """
        assert mode in ("train", "test", "auto")
        print("{} search guidance {} (bilevel/imperative) in domain {}...".format(
            {"train": "Training", "test": "Loading", "auto": "Training"}[mode],
            self.__class__.__name__, train_env_name), flush=True)
        env = pddlgym.make("PDDLEnv{}-v0".format(train_env_name))
        num_problems = min(self._num_train_problems, len(env.problems))
        states = []
        for idx in range(num_problems):
            env.fix_problem_index(idx)
            state, _ = env.reset()
            if type(state.goal).__name__ == "Literal":
                state = state.with_goal(LiteralConjunction([state.goal]))
            states.append(state)

        self._setup_feature_indices(states)

        # A dummy dataset[0] just to let setup_graph_net infer dimensions,
        # mirroring GNNSearchGuidance._create_graph_dataset -- the "nodes"
        # target dim is 1 (a single importance score per object).
        graph_input, node_to_objects = self._state_to_graph(states[0])
        dummy_target = {
            "n_node": graph_input["n_node"], "n_edge": graph_input["n_edge"],
            "edges": graph_input["edges"], "senders": graph_input["senders"],
            "receivers": graph_input["receivers"], "globals": graph_input["globals"],
            "nodes": np.zeros((len(node_to_objects), 1), dtype=np.int64),
        }
        dummy_dataset = [{"graph_input": graph_input, "graph_target": dummy_target}]
        self._model = setup_graph_net(dummy_dataset, use_gpu=False, num_steps=3)

        model_outfile = self._save_model_prefix + "_{}.pt".format(train_env_name)

        if mode == "test":
            assert os.path.exists(model_outfile), (
                "--test requires a saved checkpoint at {}, but none exists. "
                "Run with --train first.".format(model_outfile))
            self._model.load_state_dict(torch.load(model_outfile))
            print("[test mode] Loaded checkpoint from {}.".format(model_outfile), flush=True)
            return

        if mode == "auto" and self._load_from_file and os.path.exists(model_outfile):
            self._model.load_state_dict(torch.load(model_outfile))
            print("Loaded saved model from {}.".format(model_outfile), flush=True)
            return

        # mode == "train", or mode == "auto" with no cached checkpoint: run
        # the online bilevel loop from scratch.
        optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)
        pos_weight = self._bce_pos_weight * torch.ones([1])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        def base_planner_factory():
            return TrackedFD(alias_flag="--alias lama-first")

        planner = IFlaxPlanner(
            is_strips_domain=True, base_planner_factory=base_planner_factory,
            search_guider=self, seed=self._seed, gamma=self._gamma,
            complementary_rules=self._complementary_rules,
            relaxation_rules=self._relaxation_rules,
            expansion_frac=self._expansion_frac, mode="train")

        history = []
        for epoch in range(self._num_epochs):
            epoch_start = time.time()
            solved, failed = 0, 0
            running_loss = 0.0
            for state in states:
                self._model.train(True)
                # Lower level: prune with current weights, plan (+3R recovery).
                self._refresh_score_cache(state)
                try:
                    plan = planner(env.domain, state, timeout)
                except (PlanningFailure, PlanningTimeout):
                    failed += 1
                    continue
                solved += 1

                # Upper level: BCE of the ORIGINAL prediction against the
                # pseudo-label the plan just gave us (Eq. 2). Recomputed
                # with grad tracking -- the no-grad cache above is only for
                # pruning decisions during search, not backprop.
                logits, node_to_objects = self._forward(state)
                objects_in_plan = {o for act in plan for o in act.variables}
                target = torch.zeros_like(logits)
                for n, o in node_to_objects.items():
                    if o in objects_in_plan:
                        target[n, 0] = 1.0

                optimizer.zero_grad()
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print("Epoch {}/{}: solved {}/{} tasks, avg loss {:.4f}, {:.1f}s".format(
                epoch, self._num_epochs - 1, solved, len(states),
                running_loss / max(solved, 1), time.time() - epoch_start), flush=True)
            history.append({"epoch": epoch, "solved": solved, "failed": failed,
                             "loss": running_loss / max(solved, 1)})

        torch.save(self._model.state_dict(), model_outfile)
        print("Saved model to {}.".format(model_outfile), flush=True)
        if self._log_file:
            with open(self._log_file, "wb") as f:
                pickle.dump(history, f)

    def score_object(self, obj, state):
        if state != self._last_processed_state:
            self._refresh_score_cache(state)
        return self._last_object_scores[obj]
