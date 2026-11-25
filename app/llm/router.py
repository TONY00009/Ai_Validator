import json
import logging
from typing import Literal, TypeVar

from google import genai
from google.genai import types as genai_types
from groq import Groq
from openai import OpenAI
from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
ModelRole = Literal["extractor", "verifier", "eval_judge"]
ProviderName = Literal["openai", "gemini", "groq"]


class LLMRouter:
    """Central switchboard with automatic multi-model fallback routing."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._openai: OpenAI | None = None
        self._gemini: genai.Client | None = None
        self._groq: Groq | None = None

    # --- Lazy Client Initializers ---
    @property
    def openai_client(self) -> OpenAI:
        if self._openai is None:
            if not self.settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is missing in .env")
            self._openai = OpenAI(api_key=self.settings.openai_api_key)
        return self._openai

    @property
    def gemini_client(self) -> genai.Client:
        if self._gemini is None:
            if not self.settings.gemini_api_key:
                raise ValueError("GEMINI_API_KEY is missing in .env")
            self._gemini = genai.Client(api_key=self.settings.gemini_api_key)
        return self._gemini

    @property
    def groq_client(self) -> Groq:
        if self._groq is None:
            if not self.settings.groq_api_key:
                raise ValueError("GROQ_API_KEY is missing in .env")
            self._groq = Groq(api_key=self.settings.groq_api_key)
        return self._groq

    # --- Fallback Chains Per Role ---
    def get_fallback_chain(self, role: ModelRole) -> list[tuple[ProviderName, str]]:
        """
        Returns an ordered list of (provider, model_name) candidates for a role.
        Verifier stays strictly non-OpenAI to preserve cross-family verification.
        """
        if role == "extractor":
            candidates: list[tuple[ProviderName, str]] = [
                ("openai", self.settings.openai_model),
                ("groq", self.settings.groq_model),
            ]
        elif role == "verifier":
            candidates = [
                ("gemini", self.settings.gemini_model),
                *[("gemini", m) for m in self.settings.gemini_fallback_models],
                ("groq", self.settings.groq_model),
            ]
        elif role == "eval_judge":
            candidates = [
                ("groq", self.settings.groq_model),
                ("gemini", self.settings.gemini_model),
                ("openai", self.settings.openai_model),
            ]
        else:
            raise ValueError(f"Unsupported model role: {role}")

        # Remove duplicates while keeping exact priority order
        return list(dict.fromkeys(candidates))

    # --- Main Entrypoint With Automatic Failover ---
    def generate_structured(
        self,
        role: ModelRole,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        temperature: float = 0.0,
    ) -> tuple[T, str]:
        """
        Tries each model in the role's fallback chain until one succeeds.
        Returns (validated_pydantic_object, actual_model_used).
        """
        errors: list[str] = []

        for provider, model_name in self.get_fallback_chain(role):
            model_label = f"{provider}/{model_name}"
            try:
                result = self._call_provider_with_retry(
                    provider=provider,
                    model_name=model_name,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_model=response_model,
                    temperature=temperature,
                )
                return result, model_label
            except Exception as exc:
                logger.warning(
                    "⚠️  [%s] %s failed (%s). Switching to next fallback...",
                    role,
                    model_label,
                    type(exc).__name__,
                )
                errors.append(f"{model_label}: {exc}")

        raise RuntimeError(
            f"All fallback models failed for role '{role}': " + " | ".join(errors)
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=3),
        retry=retry_if_exception_type((Exception,)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _call_provider_with_retry(
        self,
        provider: ProviderName,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        temperature: float,
    ) -> T:
        if provider == "openai":
            return self._call_openai_structured(
                model_name, system_prompt, user_prompt, response_model, temperature
            )
        if provider == "gemini":
            return self._call_gemini_structured(
                model_name, system_prompt, user_prompt, response_model, temperature
            )
        if provider == "groq":
            return self._call_groq_structured(
                model_name, system_prompt, user_prompt, response_model, temperature
            )
        raise ValueError(f"Unsupported provider: {provider}")

    def _call_openai_structured(
        self,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        temperature: float,
    ) -> T:
        completion = self.openai_client.chat.completions.parse(
            model=model_name,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=response_model,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise ValueError("OpenAI returned an empty or refused structured response.")
        return parsed

    def _call_gemini_structured(
        self,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        temperature: float,
    ) -> T:
        response = self.gemini_client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                response_mime_type="application/json",
                response_schema=response_model,
            ),
        )
        if not response.text:
            raise ValueError("Gemini returned an empty response.")
        return response_model.model_validate_json(response.text)

    def _call_groq_structured(
        self,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        temperature: float,
    ) -> T:
        schema_str = json.dumps(response_model.model_json_schema(), indent=2)
        enforced_system = (
            f"{system_prompt}\n\n"
            "CRITICAL: Respond ONLY with a valid JSON object strictly matching "
            f"this JSON schema (no markdown, no extra text):\n{schema_str}"
        )
        completion = self.groq_client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": enforced_system},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw_text = completion.choices[0].message.content
        if not raw_text:
            raise ValueError("Groq returned an empty response.")
        return response_model.model_validate_json(raw_text)

    # --- Embeddings (OpenAI text-embedding-3-small) ---
    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def embed_texts(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        cleaned = [t.replace("\x00", "").strip()[:6000] or " " for t in texts]

        for i in range(0, len(cleaned), batch_size):
            batch = cleaned[i : i + batch_size]
            response = self.openai_client.embeddings.create(
                model=self.settings.openai_embedding_model,
                input=batch,
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            all_embeddings.extend([item.embedding for item in ordered])

        return all_embeddings


_router: LLMRouter | None = None


def get_llm_router() -> LLMRouter:
    global _router
    if _router is None:
        _router = LLMRouter()
    return _router