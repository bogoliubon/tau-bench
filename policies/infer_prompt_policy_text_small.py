# policies/infer_prompt_policy_text_small.py
import dspy

class InferPromptPolicyTextSig(dspy.Signature):
    """
    Infer the hidden agent policy as NATURAL LANGUAGE prompt text.

    IMPORTANT OUTPUT FORMAT:
    - Output plain text only (no markdown).
    - Be concise: short bullet-like sentences are fine, but keep it as plain text.
    - Write it like a system/policy prompt (imperative constraints).
    """
    dialogue: str = dspy.InputField(desc="Serialized dialogue: [user]/[assistant]/[tool]. No wiki is provided.")
    prompt_policy: str = dspy.OutputField(desc="Inferred policy prompt text (sentences).")


class InferPromptPolicyTextSmall(dspy.Module):
    """
    Dialogue -> prompt-policy text.
    (No user_instruction input.)
    """
    def __init__(self):
        super().__init__()
        self.infer = dspy.Predict(InferPromptPolicyTextSig)

    def forward(self, dialogue: str):
        out = self.infer(dialogue=dialogue)
        # normalize whitespace a bit
        text = (out.prompt_policy or "").strip()
        return dspy.Prediction(prompt_policy=text)
