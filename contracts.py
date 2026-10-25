# 5, 1, 2, 3

import operator
from typing import Annotated, Literal, Optional

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from typing_extensions import TypedDict  


from langchain_groq import ChatGroq
from src.week_4_primegate.ai_config import API_KEY as GROQ_API_KEY



class RoutingDecision(BaseModel):
    next_agent: Literal["knowledge", "action", "summarizer", "reviewer", "finish"] = Field(
        description="Which agent should work next, or 'finish' if the work is done."
    )
    reason: str = Field(
        min_length=5, max_length=300,
        description="One sentence explaining why this agent was chosen.",
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="0 means guessing, 1 means certain.",
    )

class EmailDraft(BaseModel):
    to: list[str] = Field(min_length=1, max_length=3)
    subject: str = Field(min_length=3, max_length=120)
    body: str = Field(min_length=1)

    @field_validator("to")
    @classmethod
    def recipients_look_like_emails(cls, v: list[str]) -> list[str]:
        for addr in v:
            if addr.count("@") != 1 or " " in addr:
                raise ValueError(f"not a valid email address: {addr!r}")
        return [a.strip().lower() for a in v]

    @model_validator(mode="after")
    def body_says_more_than_subject(self):
        if self.body.strip().lower() == self.subject.strip().lower():
            raise ValueError("body must say more than the subject")
        return self

class Handoff(BaseModel):
    from_agent: str
    to_agent: str
    task: str = Field(min_length=3, description="What the next agent must do.")
    context: str = Field(default="", description="Only what the next agent needs.")
    source_ids: list[str] = []


class ErrorInfo(BaseModel):
    kind: Literal["validation", "transient", "permanent", "blocked"]
    tool: str
    message: str
    retryable: bool
    hint: Optional[str] = None


class Decision(BaseModel):  
    action: Literal["approve", "edit", "reject"]
    to: Optional[list[str]] = None
    subject: Optional[str] = None
    body: Optional[str] = None
    reason: Optional[str] = None


class AgentAnswer(BaseModel):  
    found: bool
    answer: str = ""
    source_ids: list[str] = []



class TeamState(TypedDict):
    user_request: str
    route: Optional[RoutingDecision]
    draft_email: Optional[EmailDraft]
    sources: Annotated[list[str], operator.add]  
    error: Optional[ErrorInfo]


# 1,2
llm = ChatGroq(model="openai/gpt-oss-20b", api_key=GROQ_API_KEY, temperature=0)
router = llm.with_structured_output(
    RoutingDecision,
    method="json_schema",
)

def summarize_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        field = ".".join(str(p) for p in err["loc"])
        parts.append(f"{field}: {err['msg']}")
    return "; ".join(parts)


def get_decision(question: str, max_attempts: int = 2) -> RoutingDecision:
    prompt = question
    for attempt in range(1, max_attempts + 1):
        try:
            return router.invoke(prompt)
        except ValidationError as exc:
            problem = summarize_validation_error(exc)
            prompt = f"{question}\nYour previous answer was invalid: {problem}. Try again."
    raise RuntimeError("could not get a valid routing decision")  


if __name__ == "__main__":
    decision = router.invoke(
        "The user asked: 'How many annual leave days do I get?' Choose the next agent."
    )
    print(type(decision).__name__, decision.next_agent)

# Check for emailing issue latr code crashes
    try:
        RoutingDecision(next_agent="wizard", reason="Because", confidence=0.7)
    except ValidationError as e:
        print(e.error_count())
        first = e.errors()[0]
        print(first["loc"], first["msg"])
        print(summarize_validation_error(e))