# ADAPTER: FLAX/iFLAX write the full problem only to hand it to VAL; temporal validation
# happens on the full task inside baselines.adapter.TemporalPOPF, so nothing is written.


class PDDLProblemParser:
    @staticmethod
    def create_pddl_file(file_or_filepath, objects, initial_state, problem_name, domain_name, goal,
                         fast_downward_order=False):
        return None
