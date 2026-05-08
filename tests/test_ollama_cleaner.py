from persian_live_transcriber.ollama_cleaner import build_cleanup_prompt, build_summary_prompt


def test_cleanup_prompt_preserves_meaning_instruction() -> None:
    prompt = build_cleanup_prompt("سلام این یک تست است")
    assert "معنی" in prompt
    assert "خلاصه نکن" in prompt
    assert "سلام این یک تست است" in prompt


def test_summary_prompt_is_detailed_and_grounded() -> None:
    prompt = build_summary_prompt("جلسه درباره کیفیت ترنسکریپت فارسی بود")
    assert "چیزی اختراع نکن" in prompt
    assert "خلاصه مفصل" in prompt
    assert "## تصمیم‌ها" in prompt
    assert "## کارهای بعدی" in prompt
    assert "جلسه درباره کیفیت ترنسکریپت فارسی بود" in prompt
