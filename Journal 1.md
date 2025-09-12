## Research Concept (9/2/25 Outside of Class 30mins):

### Neural Architecture for Detecting Coordinated Inauthentic Behavior in Digital Communications

Novelty: Fusion of temporal, textual, and network signals.

Goal: build a neural architecture trained on a labeled dataset from TwiBot-20 and Russian IRA on digital communications, that can discover inauthentic communications (bot networks/communications).

Publication Points:
polyci -> Combatting Fake News 
machine learning -> Novel Fusion Method
Dataset -> kaggle + seperate repository (master dataset compiled from both)

## Sweezy Notes + Model Architecture Notes (9/3/25):

Through WHS --> Access to Nature, JSTOR & other general research libraries.

Most insightful was access to Nature, which can be very useful for researching novel (and often ground-breaking) literature.
The rest was mostly repetitive, because I have become quite familiar with library resources over the years.

### Designed Overall Architecture Flow

```mermaid
graph TD
    A[Raw Social Media Data] --> B[Multi-Modal Encoder]
    B --> C[Text Encoder<br/>BERT-based]
    B --> D[Temporal Encoder<br/>Positional Encoding]
    B --> E[Graph Encoder<br/>GCN]
    
    C --> F[Cascade Attention Module]
    D --> F
    E --> F
    
    F --> G[Micro Attention<br/>Seconds-Hours]
    F --> H[Meso Attention<br/>Hours-Days]
    F --> I[Macro Attention<br/>Days-Months]
    
    G --> J[Cross-Modal Fusion]
    H --> J
    I --> J
    
    J --> K[Hierarchical Detection]
    K --> L[Individual Bot<br/>Classifier]
    K --> M[Group Coordination<br/>Detector]
    K --> N[Campaign Orchestrator<br/>Identifier]
    
    L --> O[Final Predictions]
    M --> O
    N --> O
    
    style A fill:#e1f5fe
    style O fill:#c8e6c9
    style F fill:#fff3e0
    style J fill:#f3e5f5
```
Brushed up on NLP and Attention Mechanisms:

[Attention is All You Need](https://arxiv.org/abs/1706.03762)

[BERT: Pre-Training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

## Researched Data Sources (9/4/25)

[TwiBot-22: Towards Graph-Based Twitter Bot Detection](https://arxiv.org/abs/2206.04564)

[Russian Troll Tweets](https://github.com/fivethirtyeight/russian-troll-tweets/)

Researched viable datasets and found TwiBot-22 (Investigated for dataset completion --> looked through dataset)

## Designed Data Pipelines (9/5/25)

### Data Flow Architecture

```mermaid
graph LR
    subgraph "Input Data"
        A[Post Text]
        B[Timestamp]
        C[User Network]
        D[Metadata]
    end
    
    subgraph "Feature Extraction"
        E[BERT Embeddings<br/>768-dim]
        F[Temporal Features<br/>128-dim]
        G[Graph Features<br/>256-dim]
        H[Account Features<br/>64-dim]
    end
    
    subgraph "Attention Processing"
        I[Multi-Head Attention<br/>12 heads × 64-dim]
        J[Scale-Specific Masks]
        K[Attention Weights]
    end
    
    subgraph "Output Layers"
        L[Individual Score<br/>0-1]
        M[Group Score<br/>0-1]
        N[Campaign Score<br/>0-1]
    end
    
    A --> E
    B --> F
    C --> G
    D --> H
    
    E --> I
    F --> I
    G --> I
    H --> I
    
    I --> J
    J --> K
    K --> L
    K --> M
    K --> N
    
    style A fill:#ffebee
    style E fill:#e8f5e8
    style I fill:#fff8e1
    style L fill:#e3f2fd
```

### Training Data Structure

```mermaid
graph TB
    subgraph "Labeled Training Data"
        A[Known Bot Accounts<br/>25,000 examples]
        B[Confirmed Human Accounts<br/>50,000 examples]
        C[Coordinated Groups<br/>500 documented cases]
        D[Individual Bad Acts<br/>2,000 examples]
    end
    
    subgraph "Feature Categories"
        E[Text Features<br/>• Message content<br/>• Language patterns<br/>• Sentiment scores]
        F[Behavioral Features<br/>• Posting frequency<br/>• Response timing<br/>• Activity patterns]
        G[Network Features<br/>• Follower patterns<br/>• Interaction graphs<br/>• Clustering coefficients]
        H[Temporal Features<br/>• Account age<br/>• Post timing<br/>• Coordination windows]
    end
    
    subgraph "Label Types"
        I[Binary Labels<br/>Bot vs Human]
        J[Coordination Labels<br/>Group membership]
        K[Campaign Labels<br/>Operation affiliation]
    end
    
    A --> E
    A --> F
    A --> G
    A --> H
    
    B --> E
    B --> F
    B --> G
    B --> H
    
    E --> I
    F --> I
    G --> J
    H --> K
    
    style A fill:#ffcdd2
    style B fill:#c8e6c9
    style E fill:#e1f5fe
    style I fill:#f3e5f5
```
## Designed Attention Mechanism and Scoped More General Pipeline (9/5/25 ~1.5 hours Outside of Class)

[Attention is All You Need](https://arxiv.org/abs/1706.03762)
### Cascade Attention Mechanism

```mermaid
graph TD
    subgraph "Input Sequence"
        A[Post 1<br/>t=0:00]
        B[Post 2<br/>t=0:05]
        C[Post 3<br/>t=1:30]
        D[Post 4<br/>t=25:00]
    end
    
    subgraph "Micro Attention (0-1 hour)"
        E[Attention Matrix<br/>Local Patterns]
        F[High attention between<br/>Posts 1,2,3]
    end
    
    subgraph "Meso Attention (1-24 hours)"
        G[Attention Matrix<br/>Medium Patterns] 
        H[Medium attention to<br/>Post 4]
    end
    
    subgraph "Macro Attention (1-30 days)"
        I[Attention Matrix<br/>Long Patterns]
        J[Campaign-level<br/>Coordination]
    end
    
    A --> E
    B --> E
    C --> E
    D --> G
    
    E --> F
    G --> H
    I --> J
    
    F --> K[Fused Representation]
    H --> K
    J --> K
    
    style E fill:#ffcdd2
    style G fill:#c8e6c9
    style I fill:#bbdefb
    style K fill:#f8bbd9
```

### Detection Pipeline

```mermaid
flowchart TD
    A[Raw Social Media Posts] --> B{Preprocessing}
    
    B --> C[Text Cleaning<br/>• Remove URLs<br/>• Normalize mentions<br/>• Extract hashtags]
    B --> D[Temporal Processing<br/>• Parse timestamps<br/>• Calculate intervals<br/>• Group by windows]
    B --> E[Graph Construction<br/>• Build user networks<br/>• Extract communities<br/>• Calculate centrality]
    
    C --> F[Feature Engineering]
    D --> F
    E --> F
    
    F --> G[Model Input<br/>Tensor Shape:<br/>batch_size × seq_len × feature_dim]
    
    G --> H[Cascade Attention<br/>Multi-scale Analysis]
    
    H --> I{Threshold Check}
    
    I -->|Score > 0.8| J[High Confidence<br/>Coordinated Behavior]
    I -->|0.3 < Score < 0.8| K[Manual Review<br/>Required]
    I -->|Score < 0.3| L[Likely Authentic<br/>Behavior]
    
    J --> M[Alert/Flag System]
    K --> N[Human Analyst]
    L --> O[No Action]
    
    style A fill:#e3f2fd
    style H fill:#fff3e0
    style J fill:#ffcdd2
    style L fill:#c8e6c9
```
## Mathematical Modeling (9/8/25 Outside of Class (2 hours)

### Mathematical Attention Visualization

#### Attention Weight Heatmap Example

```
Time →
Users ↓    0h    1h    6h   24h   48h
User A   [1.0] [0.9] [0.2] [0.1] [0.0]
User B   [0.9] [1.0] [0.2] [0.1] [0.0]  ← High coordination
User C   [0.9] [0.9] [1.0] [0.1] [0.0]  ← with Users A,B
User D   [0.1] [0.1] [0.2] [1.0] [0.8]  ← Different pattern
User E   [0.2] [0.3] [0.4] [0.5] [1.0]  ← Natural variation
```

### Coordination Score Formula
```
Coordination Score = Σ(temporal_sync × content_similarity × network_overlap)

Where:
• temporal_sync = exp(-(time_diff)²/σ²)
• content_similarity = cosine_similarity(text_embeddings)
• network_overlap = |common_connections| / |total_connections|
```

## Validation Methods + Success Metrics + Proposed Workflow(9/9/25)

### Training Data Sources and Labels

#### Ground Truth Hierarchy

```mermaid
graph TD
    A[Ground Truth Sources] --> B[High Confidence Labels]
    A --> C[Medium Confidence Labels]
    A --> D[Uncertain Labels]
    
    B --> E[Platform-Banned Bots<br/>99% confidence]
    B --> F[Identical Content<br/>95% confidence]
    B --> G[Documented Campaigns<br/>90% confidence]
    
    C --> H[Template-like Content<br/>70% confidence]
    C --> I[Suspicious Timing<br/>60% confidence]
    C --> J[Network Anomalies<br/>65% confidence]
    
    D --> K[Borderline Cases<br/>30-50% confidence]
    D --> L[Human Review Needed<br/>Requires expert judgment]
    
    style E fill:#c8e6c9
    style F fill:#c8e6c9
    style G fill:#c8e6c9
    style H fill:#fff3e0
    style K fill:#ffcdd2
```

#### Cross-Validation Strategy
```mermaid
graph LR
    A[Full Dataset] --> B[Training 70%]
    A --> C[Validation 15%]
    A --> D[Test 15%]
    
    B --> E[Known Bots<br/>Known Humans<br/>Coordination Groups]
    
    C --> F[Hyperparameter<br/>Tuning]
    
    D --> G[Final Performance<br/>Evaluation]
    
    style B fill:#e8f5e8
    style C fill:#fff3e0
    style D fill:#ffebee
```

### Model Performance Metrics

#### Evaluation Framework

| Metric | Target | Reasoning |
|--------|--------|-----------|
| **Precision** | >85% | Minimize false accusations |
| **Recall** | >75% | Catch most coordinated behavior |
| **F1-Score** | >80% | Balanced performance |
| **False Positive Rate** | <5% | Protect legitimate users |
| **Detection Speed** | <100ms | Real-time capability |

### Implementation Notes
**Workflow**:
1. **Download TwiBot-20** (largest, most comprehensive)
2. **Add Russian IRA data** (political coordination examples)
3. **Supplement with Reddit bot lists** (platform diversity)
4. **Train initial model** (you now have 250K+ labeled examples)
5. **Validate on recent events** (apply to unlabeled recent data)
