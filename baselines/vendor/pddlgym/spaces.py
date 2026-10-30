# ADAPTER: FLAX/iFLAX build a LiteralSpace only to ground action literals for non-operator
# domains; temporal domains are always operators-as-actions, so it is never queried.


class LiteralSpace:
    def __init__(self, predicates, type_to_parent_types=None, **kwargs):
        self.predicates = predicates

    def all_ground_literals(self, state, valid_only=True):
        return set()
