from persian_live_transcriber.ollama_cleaner import build_cleanup_prompt, build_summary_prompt


def test_cleanup_prompt_requests_meaning_aware_repair() -> None:
    prompt = build_cleanup_prompt("من امروز meeting دارم با تیما deploy")
    assert "از نظر معنایی قابل فهم" in prompt
    assert "محافظه‌کارانه حدس بزن" in prompt
    assert "جمله‌ای معنی‌دار" in prompt
    assert "اطلاعات تازه" in prompt
    assert "کلمات و مخفف‌های انگلیسی را با حروف لاتین حفظ کن" in prompt
    assert "API" in prompt
    assert "meeting" in prompt
    assert "deploy" in prompt
    assert "خلاصه نکن" in prompt
    assert "توضیح نده" in prompt
    assert "Markdown اضافه نکن" in prompt
    assert "من امروز meeting دارم با تیما deploy" in prompt


def test_summary_prompt_is_detailed_and_grounded() -> None:
    prompt = build_summary_prompt("جلسه درباره کیفیت ترنسکریپت فارسی بود")
    assert "چیزی اختراع نکن" in prompt
    assert "خلاصه مفصل" in prompt
    assert "## تصمیم‌ها" in prompt
    assert "## کارهای بعدی" in prompt
    assert "جلسه درباره کیفیت ترنسکریپت فارسی بود" in prompt
