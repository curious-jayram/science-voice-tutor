import warnings

from google.genai import _extra_utils
from google.genai.types import GenerateContentConfig

from pipecat.adapters.services.gemini_adapter import GeminiLLMAdapter
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.google.llm import GoogleLLMService

from bot import split_spoken_citations, strip_file_search_citations


def test_strip_file_search_citations_from_spoken_sentences():
    spoken = (
        "Electricity is Chapter 11 in your NCERT textbook [PerQueryResult(index='1.2')]. "
        "In wires, current flows from the positive terminal "
        "[PerQueryResult(index='1.1'), PerQueryResult(index='1.3')]. "
        "Would you like Ohm's Law [PerQueryResult(index='1.4'), PerQueryResult(index='1.5')]?"
    )

    assert strip_file_search_citations(spoken) == (
        "Electricity is Chapter 11 in your NCERT textbook. "
        "In wires, current flows from the positive terminal. "
        "Would you like Ohm's Law?"
    )


def test_split_spoken_citations_holds_an_unfinished_marker():
    emit, held = split_spoken_citations("textbook [Per")

    assert emit == "textbook"
    assert held == " [Per"

    emit, held = split_spoken_citations(held + "QueryResult(index='1.2')]. Next")

    assert emit == ". Next"
    assert held == ""


def test_split_spoken_citations_leaves_ordinary_text_alone():
    assert split_spoken_citations("The unit is the ampere.") == (
        "The unit is the ampere.",
        "",
    )
    assert split_spoken_citations("Perfect, let's start.") == (
        "Perfect, let's start.",
        "",
    )


def test_system_instruction_is_not_a_context_message():
    adapter = GeminiLLMAdapter()
    context = LLMContext(messages=[{"role": "user", "content": "What is current?"}])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        params = adapter.get_llm_invocation_params(
            context, system_instruction="You are an NCERT Science tutor."
        )

    assert params["system_instruction"] == "You are an NCERT Science tutor."
    assert params["messages"][0].role == "user"
    assert not any("LLMContext" in str(warning.message) for warning in caught)


def test_file_search_disables_automatic_function_calling():
    llm = GoogleLLMService(
        api_key="test",
        settings=GoogleLLMService.Settings(
            model="gemini-3.8-flash",
            extra={"automatic_function_calling": {"disable": True}},
        ),
    )
    params = llm._build_generation_params(
        system_instruction="You are an NCERT Science tutor.",
        tools=[{"file_search": {"file_search_store_names": ["fileSearchStores/ncert"]}}],
    )
    config = GenerateContentConfig(**params)

    assert config.system_instruction == "You are an NCERT Science tutor."
    assert config.tools[0].file_search.file_search_store_names == ["fileSearchStores/ncert"]
    assert _extra_utils.should_disable_afc(config) is True
