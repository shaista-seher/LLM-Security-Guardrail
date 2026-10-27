
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from guardrail import ModernBERTGuardrail


guardrail = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global guardrail
    guardrail = ModernBERTGuardrail()
    yield
    guardrail = None


app = FastAPI(
    title="ModernBERT Security Guardrail API",
    description="Classifies prompts and contextual content for security review.",
    version="1.0.0",
    lifespan=lifespan,
)


class GuardrailRequest(BaseModel):
    user_prompt: str = Field(min_length=1, max_length=20000)
    conversation_history: str = Field(default="", max_length=20000)
    retrieved_context: str = Field(default="", max_length=20000)
    tool_output: str = Field(default="", max_length=20000)
    memory: str = Field(default="", max_length=20000)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "ModernBERT",
        "loaded": guardrail is not None,
    }


@app.post("/guardrail")
def check_prompt(request: GuardrailRequest):
    if guardrail is None:
        raise HTTPException(
            status_code=503,
            detail="Guardrail model is not ready.",
        )

    try:
        result = guardrail.evaluate(
            user_prompt=request.user_prompt,
            conversation_history=request.conversation_history,
            retrieved_context=request.retrieved_context,
            tool_output=request.tool_output,
            memory=request.memory,
        )
        return result

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
