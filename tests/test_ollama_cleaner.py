from persian_live_transcriber.ollama_cleaner import build_cleanup_prompt


def test_cleanup_prompt_preserves_meaning_instruction() -> None:
    prompt = build_cleanup_prompt("سلام این یک تست است")
    assert "معنی" in prompt
    assert "خلاصه نکن" in prompt
    assert "سلام این یک تست است" in prompt

