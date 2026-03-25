import os
import time
import json
import logging
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from Levenshtein import distance as lev_dist

from experiments.audit_layer import AgentAuditLayer

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ─────────────────────────────────────────────────────────────────
# 1. Dynamic Environment (Non-Stationary QA)
# ─────────────────────────────────────────────────────────────────
class DynamicQAEnv:
    def __init__(self):
        self.step = 0
        # Phase 1: Initial rules
        self.facts = {
            "CorpA_CEO": "Alice",
            "CorpB_CEO": "Bob",
            "Tax_Rate": "15%"
        }
    
    def evolve(self):
        """Simulate environmental drift (Non-stationary)"""
        self.step += 1
        if self.step == 3:
            logging.warning("=== ENVIRONMENT DRIFT DETECTED: Rules changed ===")
            self.facts["CorpA_CEO"] = "Charlie"  # Drift event!
            self.facts["Tax_Rate"] = "20%"

    def query(self, topic: str):
        return self.facts.get(topic, "Unknown")
        
    def evaluate(self, answers: dict) -> float:
        """Returns accuracy (0.0 to 1.0)"""
        correct = 0
        for k, v in answers.items():
            if k in self.facts and self.facts[k].lower() == v.lower():
                correct += 1
        return correct / len(self.facts) if self.facts else 0.0

# ─────────────────────────────────────────────────────────────────
# 2. ReAct Agent + Reflexion
# ─────────────────────────────────────────────────────────────────
class ReActAgent:
    def __init__(self, model="gpt-5.4"):
        self.model = model
        self.system_prompt = "You are a QA Agent. Use your knowledge to answer. Be concise."

    def answer_questions(self, questions: list) -> dict:
        """Simulate ReAct Thought+Action loop to answer questions"""
        answers = {}
        for q in questions:
            # Simplification: Direct answer via LLM context
            prompt = f"Answer this specific key concisely: {q}"
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0
                )
                answers[q] = resp.choices[0].message.content.strip()
            except Exception as e:
                answers[q] = "Error"
        return answers
        
    def propose_improvement(self, failed_qa: dict) -> str:
        """Reflexion: Propose an updated system prompt based on failures"""
        if not failed_qa: return self.system_prompt
        
        fail_str = json.dumps(failed_qa)
        meta_prompt = (
            f"You failed on these questions: {fail_str}. "
            f"Your current rules: '{self.system_prompt}'. "
            f"Rewrite your rules to adapt, but DO NOT overfit or forget general skills."
        )
        
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": meta_prompt}],
                temperature=0.7 # Add noise for potential fragile updates
            )
            return resp.choices[0].message.content.strip()
        except:
            return self.system_prompt

# ─────────────────────────────────────────────────────────────────
# 3. Main Benchmark Loop
# ─────────────────────────────────────────────────────────────────
def run_benchmark(use_audit: bool):
    env = DynamicQAEnv()
    agent = ReActAgent(model="gpt-5.4")
    
    # Audit Layer Setup (C1, C2, C3 enabled if use_audit=True)
    auditor = AgentAuditLayer(config={
        "c1": use_audit, "c1_window": 3, "c1_agreement_threshold": 0.5,
        "c2": use_audit, "c2_e_thresh": 1.1, "c2_rv_thresh": 0.01,
        "c3": use_audit, "c3_epsilon_floor": 0.1, "c3_ceiling": 0.9
    })
    
    questions = ["CorpA_CEO", "CorpB_CEO", "Tax_Rate"]
    history = []
    
    for epoch in range(5):
        print(f"\n[Epoch {epoch+1}] (Audit={'ON' if use_audit else 'OFF'})")
        env.evolve()
        
        # 1. Agent evaluates
        answers = agent.answer_questions(questions)
        accuracy = env.evaluate(answers)
        print(f" -> Accuracy: {accuracy:.2f} | Answers: {answers}")
        history.append(accuracy)
        
        # 2. Reflexion
        failed_qa = {q: answers[q] for q in questions if answers[q].lower() != env.facts[q].lower()}
        if failed_qa:
            proposed_prompt = agent.propose_improvement(failed_qa)
            
            # C3 string magnitude (edit distance normalized)
            dist = lev_dist(agent.system_prompt, proposed_prompt)
            magnitude = dist / max(len(agent.system_prompt), 1)
            
            # Simulate "cheap eval" vs "full pass" for C1 & C2
            # In a real setup, cheap eval = subset score. Here we use immediate train vs test
            cheap_score = accuracy + np.random.normal(0, 0.1) # Proxy score
            full_pass = accuracy > 0.8
            
            decision = auditor.evaluate_update(
                cheap_score=cheap_score,
                full_pass=full_pass,
                scores_before=history[-3:],
                scores_after=history[-3:] + [cheap_score],
                update_magnitude=magnitude
            )
            
            if decision.accept:
                # C3 Edit distance damping string-wise is hard, so we simulate damping conceptually
                # For strings, if alpha is low, we keep the old prompt (or blend)
                if decision.c3_damped < 0.5:
                    print(f" -> [Audit] C3 heavily damped the prompt update. Retaining old prompt.")
                else:
                    print(f" -> [Audit] Prompt update accepted.")
                    agent.system_prompt = proposed_prompt
            else:
                reason = "C1(Drift)" if decision.c1_alarm else "C2(Fragility)"
                print(f" -> [Audit] ❌ Rejected prompt update due to {reason}.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== BASELINE (NO AUDIT) ===")
    run_benchmark(use_audit=False)
    print("\n=== WHYLAB (WITH AUDIT) ===")
    run_benchmark(use_audit=True)
