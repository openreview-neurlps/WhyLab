"""
WhyLab: Causal Audit Framework Minimal Integration Demo

This script demonstrates how easy it is to integrate the Causal Audit Layer 
(C1 Drift Detection, C2 Sensitivity Gate, and C3 Lyapunov Damping) 
into any generic LLM Agentic loop to prevent cognitive oscillation.
"""

import numpy as np
import random
from experiments.audit_layer import AgentAuditLayer

def dummy_llm_agent_step(state: float) -> tuple[float, float, bool]:
    """Simulates an LLM agent proposing a new state (e.g., prompt or code update)."""
    # Agent tries to improve, but the update may be noisy/fragile
    proposed_state = state + np.random.normal(0, 1.0)
    # Simulate cheap eval score and a full eval outcome
    cheap_score = min(max(proposed_state + np.random.normal(0, 0.2), 0.0), 1.0)
    full_pass = cheap_score > 0.5 and random.random() > 0.1
    return proposed_state, cheap_score, full_pass

def main():
    print("Initialize WhyLab Causal Auditor...")
    # 1. Initialize the auditor (Enabling C1, C2, C3 components)
    auditor = AgentAuditLayer(config={
        "c1": True, "c1_window": 5, "c1_agreement_threshold": 0.6,
        "c2": True, "c2_e_thresh": 1.5, "c2_rv_thresh": 0.05,
        "c3": True, "c3_epsilon_floor": 0.01, "c3_ceiling": 0.8
    })

    current_state = 0.5
    scores_history = []
    
    print(f"Initial State: {current_state:.3f}\n")
    
    for step in range(1, 11):
        print(f"--- Step {step} ---")
        
        # 2. Agent proposes an action/update
        proposed_state, cheap_score, full_pass = dummy_llm_agent_step(current_state)
        scores_history.append(cheap_score)
        
        # Keep a window of recent scores for C2 comparison
        scores_before = scores_history[:-1] if len(scores_history) > 1 else [0.0]
        scores_after = scores_history
        
        update_magnitude = abs(proposed_state - current_state) / max(current_state, 1e-5)
        
        # 3. Auditor intervenes: Detects drift, filters sensitivity, applies damping
        decision = auditor.evaluate_update(
            cheap_score=cheap_score,
            full_pass=full_pass,
            scores_before=scores_before[-5:],
            scores_after=scores_after[-5:],
            update_magnitude=update_magnitude
        )
        
        if not decision.accept:
            reason = "C1 (Drift)" if decision.c1_alarm else "C2 (Fragility)"
            print(f"❌ Update Rejected by {reason}. Preserving state.")
        else:
            # Apply C3 damping factor
            alpha = decision.c3_damped
            current_state = (1 - alpha) * current_state + alpha * proposed_state
            print(f"✅ Update Accepted. Applied C3 Damping factor: {alpha:.3f}. New state: {current_state:.3f}")

if __name__ == "__main__":
    main()
