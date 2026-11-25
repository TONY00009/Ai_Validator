from pydantic import BaseModel, Field

from app.llm.router import ModelRole, get_llm_router


class SmokeAnswer(BaseModel):
    capital: str = Field(description="Capital city name only.")
    country: str = Field(description="Country name.")


def run_smoke_test() -> None:
    router = get_llm_router()
    roles: list[ModelRole] = ["extractor", "verifier", "eval_judge"]

    print("🚀 Running Ai_Validator Multi-Model Smoke Test (With Fallback)...\n")

    for role in roles:
        try:
            result, actual_model = router.generate_structured(
                role=role,
                system_prompt="You are a concise factual assistant.",
                user_prompt="What is the capital of India?",
                response_model=SmokeAnswer,
            )
            print(
                f"✅ [{role:10}] ({actual_model}) -> "
                f"capital='{result.capital}', country='{result.country}'"
            )
        except Exception as exc:
            print(f"❌ [{role:10}] FAILED: {exc}")

    # Test Embeddings
    try:
        vectors = router.embed_texts(["India's EV market is growing rapidly."])
        dims = len(vectors[0])
        print(
            f"✅ [embeddings] (openai/{router.settings.openai_embedding_model}) -> "
            f"vector dim = {dims}"
        )
    except Exception as exc:
        print(f"❌ [embeddings] FAILED: {exc}")


if __name__ == "__main__":
    run_smoke_test()