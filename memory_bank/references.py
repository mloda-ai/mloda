from enum import Enum


class MemoryBankReferences(Enum):
    MEMORY_BANK = "memory_bank"
    LLMGENERATED_CONTENT = f"{MEMORY_BANK}/llm_generated_content"
    PROJECTBRIEF = f"{LLMGENERATED_CONTENT}/projectbrief.md"
    MEMORY_BANK_PROMPT = "memory_bank_prompt.md"
    README = f"{MEMORY_BANK}/README.md"
    DOCS = f"{MEMORY_BANK}/docs"
    MLODA_CORE = f"{MEMORY_BANK}/mloda_core"
    TESTS = f"{MEMORY_BANK}/tests"
    MLODA_PLUGINS = f"{MEMORY_BANK}/mloda_plugins"
    MEMORY_BANK_PROMPTS = f"{MEMORY_BANK}/prompts"
