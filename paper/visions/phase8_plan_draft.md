
## Phase 8: The Living Ledger (Cellular Agent Architecture)
> **Goal**: 단세포 분석 도구에서 자율적인 인과추론 유기체로 진화합니다.

### 1. Nucleus Upgrade: Discovery Agent (`engine/agents/discovery.py`)
- **Objective**: "무엇이 무엇의 원인인가?"를 스스로 발견하는 지능 구현.
- **Tech Stack**: `causal-learn`, `OpenAI API / Gemini API`
- **Logic**:
    1.  **Schema Analysis**: LLM이 테이블 컬럼명과 설명을 읽고 상식적인 인과관계 가설 수립 (Prior Knowledge).
    2.  **Statistical Discovery**: PC(Peter-Clark) 알고리즘으로 데이터의 조건부 독립성 검정.
    3.  **Hybrid DAG**: LLM의 가설과 통계적 결과를 결합하여 최종 DAG 생성.

### 2. Membrane: MCP Server (`engine/server/mcp_server.py`)
- **Objective**: WhyLab 엔진을 외부 세계와 연결하는 표준 인터페이스 구축.
- **Tech Stack**: `mcp-python-sdk`
- **Resources**:
    -   `whylab://data/latest`: 최신 분석 결과 JSON.
    -   `whylab://reports/white_paper`: 백서 Markdown.
- **Tools**:
    -   `run_analysis(scenario: str)`: 특정 시나리오 분석 실행.
    -   `simulate_intervention(treatment: str, value: float)`: What-If 시뮬레이션.
- **Prompts**:
    -   `analyze_causality`: "이 데이터셋에서 인과관계를 분석해줘" 요청 처리.

### 3. Cytoplasm: LangGraph Workflow (`engine/workflow/graph.py`)
- **Objective**: 에이전트 간의 순환적(Cyclic) 협업 및 상태 관리.
- **Tech Stack**: `langgraph`, `langchain`
- **State**:
    ```python
    class AgentState(TypedDict):
        data: pd.DataFrame
        dag: list[tuple]
        hypothesis: str
        confidence: float
        history: list[str]
    ```
- **Nodes**:
    -   `DiscoveryNode`: 가설 수립 및 DAG 생성.
    -   `EstimationNode`: CATE 추정 및 정책 도출.
    -   `RefutationNode`: Placebo 테스트 수행.
- **Edges**:
    -   `check_integrity`: Refutation 실패 시 `DiscoveryNode`로 회귀 (Feedback Loop).

### 4. Tissue Formation: Orchestration (`experiments/tissue_simulation.py`)
- **Objective**: 다중 에이전트가 협력하여 복잡한 문제를 해결하는 시뮬레이션.
- **Scenario**:
    -   데이터 드리프트 발생(User Behavior Change) 상황 주입.
    -   기존 모델의 Refutation 실패 감지.
    -   Discovery Agent가 새로운 교란 변수 탐지 및 DAG 수정.
    -   Estimation Agent가 모델 재학습 및 리포트 갱신.
