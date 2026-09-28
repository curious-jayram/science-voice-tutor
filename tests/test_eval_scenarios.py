from pathlib import Path

from pipecat.evals.scenario import EvalKind, EvalScenarioFile
from pipecat.evals.script import EvalScriptScenario
from pipecat.evals.simulation import EvalSimulationScenario
from pipecat.evals.suite import EvalManifest

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = ROOT / "scenarios"


def test_scripted_and_simulated_scenarios_load():
    loaded = [
        EvalScenarioFile.load(path)
        for path in sorted(SCENARIOS.rglob("*.yaml"))
        if path.parent != SCENARIOS
    ]

    by_file = {scenario_file.name: scenario_file for scenario_file in loaded}
    assert set(by_file) == {"photosynthesis", "out_of_book", "class_8_study", "audio"}

    photosynthesis = by_file["photosynthesis"].scenarios[0]
    assert isinstance(photosynthesis, EvalScriptScenario)
    assert photosynthesis.name == "photosynthesis/class_8"
    assert photosynthesis.turns[1].user.startswith("I am in class 8")
    assert photosynthesis.turns[1].expect[0].text_excludes == "PerQueryResult"

    refusal = by_file["out_of_book"].scenarios[0]
    assert isinstance(refusal, EvalScriptScenario)
    assert "quantum field theory" in refusal.turns[1].user

    study = by_file["class_8_study"].scenarios[0]
    assert isinstance(study, EvalSimulationScenario)
    assert study.name == "class_8_study/photosynthesis"
    assert study.runs == 3
    assert study.user_audio is False
    assert {metric.name for metric in study.metrics} == {
        "spoken_style",
        "latin_script",
        "words",
        "latency",
    }

    spoken = by_file["audio"]
    assert [scenario.name for scenario in spoken.scenarios] == [
        "audio/photosynthesis",
        "audio/out_of_book",
        "audio/class_8_study",
    ]
    assert all(scenario.user_audio and scenario.bot_audio for scenario in spoken.scenarios)
    assert spoken.scenarios[0].user_speech["factory"] == "evals.speech.sarvam"
    assert spoken.scenarios[0].user_speech["voice"] == "kavya"
    assert spoken.scenarios[0].user_speech["language"] == "en-IN"
    assert spoken.scenarios[0].transcriber["service"] == "moonshine"
    assert spoken.scenarios[2].runs == 1


def test_suite_manifest_points_at_those_scenarios():
    manifest = EvalManifest.load(ROOT / "evals" / "manifest.yaml")

    assert manifest.concurrency == 1
    assert [(run.kind, run.scenario) for run in manifest.runs] == [
        (EvalKind.SCRIPT, "photosynthesis/class_8"),
        (EvalKind.SCRIPT, "out_of_book/refuses"),
        (EvalKind.SIMULATION, "class_8_study/photosynthesis"),
        (EvalKind.SCRIPT, "audio/photosynthesis"),
        (EvalKind.SCRIPT, "audio/out_of_book"),
        (EvalKind.SIMULATION, "audio/class_8_study"),
        (EvalKind.SIMULATION, "class_8_study/photosynthesis"),
        (EvalKind.SIMULATION, "class_8_study/photosynthesis"),
    ]
    assert manifest.runs[0].bot_path == (ROOT / "evals" / "serve.py").resolve()
    assert all(run.scenario_path.is_file() for run in manifest.runs)
