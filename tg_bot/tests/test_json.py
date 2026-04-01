from llm import _extract_json


def test_extract_json_plain():
    text = '{"audiences": ["пенсионеры"], "question": "возьмут кредит?", "ratios": [1]}'
    result = _extract_json(text)
    assert result["audiences"] == ["пенсионеры"]
    assert result["ratios"] == [1]


def test_extract_json_with_markdown_wrapper():
    text = '```json\n{"audiences": ["студенты"], "question": "откроют вклад?", "ratios": [1]}\n```'
    result = _extract_json(text)
    assert result["audiences"] == ["студенты"]


def test_extract_json_with_extra_text():
    text = 'Вот результат:\n{"audiences": ["врачи"], "question": "возьмут ипотеку?", "ratios": [1]}'
    result = _extract_json(text)
    assert result["audiences"] == ["врачи"]


def test_extract_json_invalid_returns_empty():
    result = _extract_json("не валидный json вообще")
    assert result == {}


def test_extract_json_empty_string():
    result = _extract_json("")
    assert result == {}
