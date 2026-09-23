"""
Forward-Chaining Inference Engine for Crop Disease Expert System.

Processes facts (disease detected, severity, environmental conditions)
against rules in the knowledge base to generate risk assessments
and context-aware recommendations.
"""


class Fact:
    """Represents a single fact in the expert system."""

    def __init__(self, name, value, source='user'):
        self.name = name
        self.value = value
        self.source = source  # 'ai', 'user', 'inferred'

    def __repr__(self):
        return f"Fact({self.name}={self.value}, source={self.source})"


class Rule:
    """Represents a rule in the expert system."""

    def __init__(self, name, conditions, conclusions, priority=1):
        """
        Args:
            name: Rule identifier.
            conditions: List of (fact_name, operator, value) tuples.
            conclusions: List of (fact_name, value) tuples to infer.
            priority: Rule priority (higher = more important).
        """
        self.name = name
        self.conditions = conditions
        self.conclusions = conclusions
        self.priority = priority
        self.fired = False

    def evaluate(self, facts_dict):
        """
        Evaluate if all conditions of this rule are met.

        Args:
            facts_dict: Dictionary mapping fact names to Fact objects.

        Returns:
            bool: True if all conditions are satisfied.
        """
        for fact_name, operator, expected_value in self.conditions:
            if fact_name not in facts_dict:
                return False

            actual_value = facts_dict[fact_name].value

            if operator == '==' and actual_value != expected_value:
                return False
            elif operator == '!=' and actual_value == expected_value:
                return False
            elif operator == '>=' and actual_value < expected_value:
                return False
            elif operator == '<=' and actual_value > expected_value:
                return False
            elif operator == '>' and actual_value <= expected_value:
                return False
            elif operator == '<' and actual_value >= expected_value:
                return False
            elif operator == 'in' and actual_value not in expected_value:
                return False
            elif operator == 'not_in' and actual_value in expected_value:
                return False

        return True


class InferenceEngine:
    """Forward-chaining inference engine."""

    def __init__(self, knowledge_base):
        """
        Args:
            knowledge_base: KnowledgeBase instance.
        """
        self.kb = knowledge_base
        self.facts = {}
        self.rules = []
        self.fired_rules = []
        self.explanations = []
        self._build_rules()

    def reset(self):
        """Reset the engine state for a new consultation."""
        self.facts = {}
        self.fired_rules = []
        self.explanations = []
        for rule in self.rules:
            rule.fired = False

    def add_fact(self, name, value, source='user'):
        """Add a fact to the working memory."""
        self.facts[name] = Fact(name, value, source)

    def _build_rules(self):
        """Build the rule set from the knowledge base."""

        # --- Severity classification rules ---
        self.rules.append(Rule(
            name='severity_healthy',
            conditions=[('severity_percentage', '<=', 5)],
            conclusions=[('severity_level', 'healthy')],
            priority=1
        ))
        self.rules.append(Rule(
            name='severity_mild',
            conditions=[('severity_percentage', '>', 5), ('severity_percentage', '<=', 20)],
            conclusions=[('severity_level', 'mild')],
            priority=1
        ))
        self.rules.append(Rule(
            name='severity_moderate',
            conditions=[('severity_percentage', '>', 20), ('severity_percentage', '<=', 40)],
            conclusions=[('severity_level', 'moderate')],
            priority=1
        ))
        self.rules.append(Rule(
            name='severity_severe',
            conditions=[('severity_percentage', '>', 40), ('severity_percentage', '<=', 60)],
            conclusions=[('severity_level', 'severe')],
            priority=1
        ))
        self.rules.append(Rule(
            name='severity_critical',
            conditions=[('severity_percentage', '>', 60)],
            conclusions=[('severity_level', 'critical')],
            priority=1
        ))

        # --- High humidity risk rules ---
        self.rules.append(Rule(
            name='high_humidity_fungal_risk',
            conditions=[
                ('humidity', '>=', 80),
                ('pathogen_type', 'in', ['Fungus', 'Oomycete'])
            ],
            conclusions=[('humidity_risk', 'high')],
            priority=2
        ))
        self.rules.append(Rule(
            name='moderate_humidity_risk',
            conditions=[
                ('humidity', '>=', 60),
                ('humidity', '<', 80)
            ],
            conclusions=[('humidity_risk', 'moderate')],
            priority=1
        ))
        self.rules.append(Rule(
            name='low_humidity_risk',
            conditions=[('humidity', '<', 60)],
            conclusions=[('humidity_risk', 'low')],
            priority=1
        ))

        # --- Temperature risk rules ---
        self.rules.append(Rule(
            name='optimal_temp_for_disease',
            conditions=[
                ('temperature_in_range', '==', True)
            ],
            conclusions=[('temperature_risk', 'high')],
            priority=2
        ))

        # --- Rainfall risk rules ---
        self.rules.append(Rule(
            name='heavy_rainfall_risk',
            conditions=[
                ('rainfall', 'in', ['heavy', 'moderate_to_heavy'])
            ],
            conclusions=[('rainfall_risk', 'high')],
            priority=2
        ))
        self.rules.append(Rule(
            name='moderate_rainfall_risk',
            conditions=[
                ('rainfall', 'in', ['moderate', 'light_to_moderate'])
            ],
            conclusions=[('rainfall_risk', 'moderate')],
            priority=1
        ))
        self.rules.append(Rule(
            name='low_rainfall_risk',
            conditions=[
                ('rainfall', 'in', ['dry', 'light'])
            ],
            conclusions=[('rainfall_risk', 'low')],
            priority=1
        ))

        # --- Combined risk escalation rules ---
        self.rules.append(Rule(
            name='critical_risk_escalation',
            conditions=[
                ('severity_level', 'in', ['severe', 'critical']),
                ('humidity_risk', '==', 'high')
            ],
            conclusions=[('risk_escalation', 'critical')],
            priority=3
        ))
        self.rules.append(Rule(
            name='high_risk_escalation',
            conditions=[
                ('severity_level', 'in', ['moderate', 'severe']),
                ('humidity_risk', 'in', ['moderate', 'high'])
            ],
            conclusions=[('risk_escalation', 'high')],
            priority=2
        ))

        # --- Urgent action rules ---
        self.rules.append(Rule(
            name='immediate_action_required',
            conditions=[
                ('severity_level', 'in', ['severe', 'critical']),
                ('confidence', '>=', 0.7)
            ],
            conclusions=[('action_urgency', 'immediate')],
            priority=3
        ))
        self.rules.append(Rule(
            name='prompt_action_required',
            conditions=[
                ('severity_level', '==', 'moderate'),
                ('confidence', '>=', 0.6)
            ],
            conclusions=[('action_urgency', 'prompt')],
            priority=2
        ))
        self.rules.append(Rule(
            name='monitoring_recommended',
            conditions=[
                ('severity_level', 'in', ['healthy', 'mild'])
            ],
            conclusions=[('action_urgency', 'monitor')],
            priority=1
        ))

        # --- Viral disease special rules ---
        self.rules.append(Rule(
            name='viral_no_cure',
            conditions=[
                ('pathogen_type', '==', 'Virus')
            ],
            conclusions=[('special_note', 'no_chemical_cure')],
            priority=3
        ))

        # --- Pest special rules ---
        self.rules.append(Rule(
            name='pest_dry_conditions',
            conditions=[
                ('pathogen_type', 'in', ['Pest (Arachnid)', 'Pest']),
                ('humidity', '<', 50)
            ],
            conclusions=[('pest_risk', 'elevated')],
            priority=2
        ))

    def run(self):
        """
        Execute forward chaining inference.

        Returns:
            dict: All inferred facts and explanations.
        """
        max_iterations = 50
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            new_fact_added = False

            # Sort rules by priority (higher first)
            sorted_rules = sorted(self.rules, key=lambda r: r.priority, reverse=True)

            for rule in sorted_rules:
                if rule.fired:
                    continue

                if rule.evaluate(self.facts):
                    rule.fired = True
                    self.fired_rules.append(rule.name)

                    for fact_name, fact_value in rule.conclusions:
                        if fact_name not in self.facts:
                            self.add_fact(fact_name, fact_value, source='inferred')
                            new_fact_added = True
                            self.explanations.append(
                                f"Rule '{rule.name}' fired: inferred {fact_name} = {fact_value}"
                            )

            if not new_fact_added:
                break

        return self._compile_results()

    def _compile_results(self):
        """Compile inference results into a structured response."""
        results = {
            'inferred_facts': {
                name: fact.value
                for name, fact in self.facts.items()
                if fact.source == 'inferred'
            },
            'all_facts': {
                name: {'value': fact.value, 'source': fact.source}
                for name, fact in self.facts.items()
            },
            'fired_rules': self.fired_rules,
            'explanations': self.explanations,
            'num_iterations': len(self.fired_rules),
        }
        return results

    def get_fact_value(self, fact_name, default=None):
        """Get the value of a fact, returning default if not found."""
        fact = self.facts.get(fact_name)
        return fact.value if fact else default
