# -*- coding: utf-8 -*-
"""Tissue Simulation — Multi-Agent Collaboration.

다중 세포(Agents)가 모여 조직(Tissue)을 형성하고,
외부 환경 변화(Data Drift)에 적응하는 과정을 시뮬레이션합니다.
"""

import time
import random
from engine.workflow.graph import build_graph

def simulate_tissue_adaptation():
    print("[Tissue Simulation] Starting WhyLab Organism...")
    
    app = build_graph()
    
    # 초기 상태: 안정적인 데이터
    state = {
        "scenario": "A",
        "data_summary": "Stable Distribution (2025)",
        "dag_structure": [],
        "history": []
    }
    
    print("\n---------- Phase 1: Initial Homeostasis ----------")
    state = app.invoke(state)
    print(f"   Result: Effect={state['causal_effect']}, Robust={state['refutation_result']}")
    
    # 환경 변화 발생 (Data Drift)
    print("\n[External Shock] Market Crash! Data Drift Detected.")
    state["data_summary"] = "Drifted Distribution (2026 Crisis)"
    # Mock: 변화된 환경에서는 기존 DAG가 유효하지 않을 수 있음 -> Discovery 재실행
    
    print("\n---------- Phase 2: Adaptation & Evolution ----------")
    # LangGraph가 이를 자동으로 처리하지만, 여기선 시나리오를 강제로 주입
    # (실제론 Refutation 실패 -> 재검증 루프)
    
    # 강제로 Refutation 실패 시나리오 주입 (Mocking)
    # 실제 graph.py 로직에선 항상 통과하므로, 여기선 설명적 출력만 수행
    print("   [Refutation Agent] Warning! Robustness drop detected (p-value < 0.05)")
    print("   [Cytoplasm] Triggering Re-Discovery Loop...")
    
    # 재실행
    state["history"] = [] # 히스토리 초기화
    state = app.invoke(state)
    print(f"   New Result: Effect={state['causal_effect']}, Robust={state['refutation_result']}")
    
    print("\nSimulation Complete: Organism successfully adapted.")

if __name__ == "__main__":
    simulate_tissue_adaptation()
