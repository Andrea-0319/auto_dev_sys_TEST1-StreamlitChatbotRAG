# Technical Design: RAG Chatbot Enhanced Features

## Architecture Overview

The enhanced RAG chatbot maintains a modular architecture with clear separation between UI, core logic, and model interaction layers. The new features extend the existing `SessionManager` to support conversation branching, add a token tracking service, and introduce a summarization engine - all while preserving backward compatibility.

```
┌─────────────────────────────────────────────────────────────────┐
│                         UI Layer (Streamlit)                     │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐│
│  │ TokenDisplay │ │ ChatControls │ │ BranchSelector/TimeTravel││
│  └──────────────┘ └──────────────┘ └──────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      Core Services Layer                          │
│  ┌──────────────────┐ ┌──────────────────┐ ┌─────────────────┐ │
│  │ SessionManager   │ │ TokenTracker     │ │ SummarizationSvc│ │
│  │ (Extended)       │ │                  │ │                 │ │
│  └──────────────────┘ └──────────────────┘ └─────────────────┘ │
│  ┌──────────────────┐ ┌──────────────────┐                     │
│  │ BranchManager    │ │ ExportService    │                     │
│  └──────────────────┘ └──────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                    Model/RAG Layer                               │
│  ┌──────────────────┐ ┌──────────────────┐ ┌─────────────────┐ │
│  │ ModelHandler     │ │ RAGEngine         │ │ PromptBuilder   │ │
│  └──────────────────┘ └──────────────────┘ └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Component Design

### 1. TokenTracker Service
**File**: `workspace/core/token_tracker.py`

**Responsibility**: Calculate and track token usage in real-time across all context components.

**Interface**:
```python
class TokenTracker:
    def __init__(self, model_handler: ModelHandler, max_context_tokens: int = 1500)
    def count_tokens(self, text: str) -> int
    def get_context_breakdown(self, messages: List[Message], system_prompt: str, context_docs: List[str]) -> TokenBreakdown
    def get_current_usage(self) -> TokenUsage
    def is_approaching_limit(self, threshold: float = 0.8) -> bool
```

**Key Classes**:
```python
@dataclass
class TokenBreakdown:
    system_prompt_tokens: int
    chat_history_tokens: int
    context_tokens: int
    total: int
    percentage: float

@dataclass
class TokenUsage:
    current: int
    max: int
    percentage: float
    breakdown: TokenBreakdown
```

### 2. Enhanced SessionManager
**File**: `workspace/core/session.py` (modified)

**New Responsibilities**:
- Manage conversation branches
- Track message metadata (pinned, feedback)
- Support branching from any message

**New Interface**:
```python
class SessionManager:
    # Existing methods (unchanged)
    
    # New branching methods
    def create_branch(self, from_message_index: int, branch_name: str = None) -> str
    def switch_branch(self, branch_id: str) -> None
    def get_current_branch_id(self) -> str
    def get_all_branches(self) -> List[Branch]
    def delete_branch(self, branch_id: str) -> bool
    
    # New message metadata methods
    def pin_message(self, message_id: str) -> bool
    def unpin_message(self, message_id: str) -> bool
    def delete_message(self, message_id: str) -> bool
    def set_feedback(self, message_id: str, feedback: str) -> bool
    
    # New search method
    def search_messages(self, query: str) -> List[Message]
```

### 3. BranchManager Service
**File**: `workspace/core/branch_manager.py`

**Responsibility**: Handle conversation branching logic and branch persistence.

**Interface**:
```python
class BranchManager:
    def __init__(self, session_manager: SessionManager)
    def create_branch(self, from_message: Message, name: str = None) -> Branch
    def list_branches(self) -> List[Branch]
    def get_branch(self, branch_id: str) -> Optional[Branch]
    def delete_branch(self, branch_id: str) -> bool
    def merge_branch(self, source_branch_id: str, target_branch_id: str) -> bool
```

### 4. SummarizationService
**File**: `workspace/core/summarization.py`

**Responsibility**: Generate context summaries using the same LLM model, replace older messages with summary.

**Interface**:
```python
class SummarizationService:
    def __init__(self, model_handler: ModelHandler)
    def summarize_messages(self, messages: List[Message], preserve_recent: int = 4) -> SummaryResult
    def create_summary_message(self, summary: str, original_message_ids: List[str]) -> Message
    def get_summary_prompt(self, messages: List[Message]) -> str
```

### 5. ExportService
**File**: `workspace/core/export.py`

**Responsibility**: Handle multiple export formats with metadata.

**Interface**:
```python
class ExportService:
    def export_markdown(self, messages: List[Message], include_metadata: bool = True) -> str
    def export_json(self, messages: List[Message], include_metadata: bool = True) -> str
    def export_plain_text(self, messages: List[Message]) -> str
    def export_with_timestamps(self, messages: List[Message], format: str = "markdown") -> str
```

### 6. Enhanced UI Components
**File**: `workspace/ui/token_display.py` (new)
**File**: `workspace/ui/branch_selector.py` (new)
**File**: `workspace/ui/chat_controls.py` (enhanced)

**Components**:
```python
# token_display.py
def render_token_indicator(usage: TokenUsage) -> None
def render_token_breakdown(breakdown: TokenBreakdown) -> None

# branch_selector.py
def render_branch_dropdown(branches: List[Branch], current_branch_id: str) -> str
def render_branch_tree(branches: List[Branch]) -> None

# chat_controls.py (enhanced)
def render_enhanced_controls(messages: List[Message]) -> None
def render_message_actions(message: Message) -> None
```

## Data Model

### Updated Message Schema
```python
@dataclass
class Message:
    role: str
    content: str
    timestamp: datetime
    reasoning: Optional[str] = None
    sources: Optional[List[str]] = None
    id: str = field(default_factory=lambda: str(uuid4()))
    # New fields
    is_pinned: bool = False
    feedback: Optional[str] = None  # "positive", "negative"
    branch_id: Optional[str] = None
    parent_message_id: Optional[str] = None  # For branching
```

### Branch Schema
```python
@dataclass
class Branch:
    id: str
    name: str
    created_at: datetime
    created_from_message_id: str
    message_count: int
    is_active: bool = True
```

### Complete Data Flow
```
User Input → ChatInput → SessionManager.add_user_message()
                                          ↓
                               TokenTracker.count_tokens()
                                          ↓
                               RAGEngine.retrieve_context()
                                          ↓
                               ModelHandler.generate()
                                          ↓
                               SessionManager.add_assistant_message()
                                          ↓
                               UI.render_chat_interface()
```

## Technology Stack

| Layer | Technology | Version/Notes |
|-------|-------------|----------------|
| UI Framework | Streamlit | Latest |
| Language | Python | 3.9+ |
| Model | Qwen/Qwen2.5-1.5B-Instruct | Configurable |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Local |
| Tokenization | HuggingFace Transformers | Built-in tokenizer |
| State Management | Streamlit Session State | Built-in |
| Serialization | JSON | Built-in |

**New Dependencies** (to add to requirements.txt):
- `tiktoken` - Fast token counting (optional, fallback to HF tokenizer)
- No new external dependencies required for core features

## File Structure
```
workspace/
├── core/
│   ├── __init__.py
│   ├── session.py           # Enhanced SessionManager
│   ├── token_tracker.py     # NEW - Token tracking
│   ├── branch_manager.py   # NEW - Branching logic
│   ├── summarization.py    # NEW - Context summarization
│   ├── export.py           # NEW - Export service
│   └── workflow.py
├── model/
│   ├── handler.py
│   ├── prompts.py
│   ├── reasoning.py
│   └── streaming.py
├── rag/
│   ├── engine.py
│   ├── vector_store.py
│   ├── embeddings.py
│   ├── chunker.py
│   └── document_processor.py
├── ui/
│   ├── app.py
│   ├── chat.py             # Enhanced
│   ├── sidebar.py
│   ├── components.py
│   ├── token_display.py    # NEW
│   ├── branch_selector.py # NEW
│   ├── chat_controls.py   # Enhanced
│   ├── document_manager.py
│   └── reasoning_panel.py
├── config/
│   └── settings.py         # Enhanced with new settings
├── utils/
│   ├── logger.py
│   ├── validators.py
│   └── file_utils.py
├── tests/
│   ├── test_token_tracker.py    # NEW
│   ├── test_branch_manager.py   # NEW
│   ├── test_summarization.py    # NEW
│   ├── test_export.py           # NEW
│   └── ... (existing)
└── requirements.txt
```

## API Design

### New Streamlit Callbacks

Since this is a Streamlit app, new features use callback patterns rather than REST APIs:

| Feature | Callback | Parameters |
|---------|----------|------------|
| Token Display | `on_token_update` | `token_usage: TokenUsage` |
| Summarize | `on_summarize` | `preserve_recent: int` |
| Create Branch | `on_create_branch` | `from_message_id: str, name: str` |
| Switch Branch | `on_switch_branch` | `branch_id: str` |
| Pin Message | `on_pin_message` | `message_id: str` |
| Feedback | `on_feedback` | `message_id: str, feedback: str` |
| Search | `on_search` | `query: str` |
| Export | `on_export` | `format: str` |

### Configuration Extensions

```python
# config/settings.py additions
@dataclass
class Config:
    # Existing settings...
    
    # New settings
    MAX_CONTEXT_TOKENS: int = 1500
    TOKEN_WARNING_THRESHOLD: float = 0.8
    SUMMARY_PRESERVE_MESSAGES: int = 4
    MAX_BRANCHES_PER_CONVERSATION: int = 10
    ENABLE_TOOLS: bool = True
    
    # Tool modes
    TOOL_MODES: dict = field(default_factory=lambda: {
        "general": {"name": "General Chat", "system_prompt": "You are a helpful assistant."},
        "code": {"name": "Code Assistant", "system_prompt": "You are an expert programmer. Focus on code quality."},
        "document": {"name": "Document Analyzer", "system_prompt": "You analyze documents and extract key information."}
    })
```

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)
1. Extend `Message` dataclass with new fields
2. Implement `TokenTracker` service
3. Add token display UI component
4. Integrate token tracking into chat flow
5. Write unit tests for TokenTracker

### Phase 2: Branching (Week 2)
1. Implement `BranchManager` service
2. Extend `SessionManager` with branching methods
3. Create branch selector UI
4. Implement branch switching logic
5. Add visual branch differentiation in chat

### Phase 3: Summarization (Week 2-3)
1. Implement `SummarizationService`
2. Add "Summarize Context" button
3. Create confirmation dialog
4. Implement message replacement logic
5. Add collapsible summary display

### Phase 4: Enhanced Controls (Week 3)
1. Add message action buttons (copy, delete, pin)
2. Implement search functionality
3. Add keyboard shortcuts
4. Implement feedback buttons

### Phase 5: Tools & Export (Week 4)
1. Add tool mode selector
2. Implement dynamic system prompt switching
3. Enhance export with JSON and plain text
4. Add timestamp inclusion in exports

### Phase 6: Polish & Testing (Week 4-5)
1. Accessibility improvements
2. Responsive design adjustments
3. Integration testing
4. Performance optimization
5. Documentation updates

## Design Decisions & Trade-offs

| Decision | Rationale |
|----------|-----------|
| Use same LLM for summarization | Meets constraint of no external APIs |
| Token counting via HF tokenizer | Ensures 5% accuracy requirement |
| Branch storage in memory | Simplicity; can persist to JSON later |
| Streamlit callbacks over REST | Matches existing architecture |
| Message deletion is soft | Allows undo; prevents data loss |
| Maximum 10 branches | Prevents memory issues |

## Security Considerations

1. **Local Processing**: All features work offline - no data leaves the system
2. **Session Isolation**: Each branch maintains separate message history
3. **No Credential Storage**: Feedback and metadata stored locally only
4. **Input Validation**: Sanitize search queries and branch names

## Testing Strategy

1. **Unit Tests**: TokenTracker accuracy, BranchManager logic, Export formats
2. **Integration Tests**: Full chat flow with branching
3. **UI Tests**: Verify all buttons and interactions work
4. **Performance Tests**: Token counting latency < 50ms
5. **Edge Cases**: Empty conversations, max branches, boundary tokens
