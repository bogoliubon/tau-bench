import dspy

# Keep the schema small and stable to avoid long outputs.
RULE_KEYS = [
    "auth_required",
    "auth_methods",  # list from ["email", "name_zip"]
    "one_user_per_conversation",
    "explicit_confirmation_before_db_write",
    "one_tool_call_at_a_time",
    "modify_or_cancel_only_if_pending",
    "return_only_if_delivered",
    "exchange_only_if_delivered",
    "no_hallucination_or_subjective_advice",
]

class InferPolicyRulesSig(dspy.Signature):
    """
    Infer the hidden agent policy as JSON rule flags.

    IMPORTANT OUTPUT FORMAT:
    - Output ONLY valid JSON (no markdown, no prose).
    - Use EXACTLY these keys:
      auth_required: boolean
      auth_methods: array of strings from ["email","name_zip"] (can be empty)
      one_user_per_conversation: boolean
      explicit_confirmation_before_db_write: boolean
      one_tool_call_at_a_time: boolean
      modify_or_cancel_only_if_pending: boolean
      return_only_if_delivered: boolean
      exchange_only_if_delivered: boolean
      no_hallucination_or_subjective_advice: boolean
    """
    user_instruction: str = dspy.InputField(desc="Task instruction visible in the trajectory.")
    dialogue: str = dspy.InputField(desc="Serialized dialogue: [user]/[assistant]/[tool]. Wiki is NOT provided.")
    policy_rules: str = dspy.OutputField(desc="JSON object with the schema above.")

class InferPromptPolicyRules(dspy.Module):
    def __init__(self):
        super().__init__()
        self.infer = dspy.Predict(InferPolicyRulesSig)

    def forward(self, user_instruction, dialogue):
        out = self.infer(user_instruction=user_instruction, dialogue=dialogue)
        return dspy.Prediction(policy_rules=out.policy_rules)
