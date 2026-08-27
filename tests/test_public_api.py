import bettermdptools.experiments as experiments


def test_experiments_exports_documented_entrypoints_and_result_types():
    assert set(experiments.__all__) == {
        "ExperimentBuilder",
        "MissingOptunaDependency",
        "OptunaResult",
        "RunResult",
        "optimize",
        "run",
    }

    assert experiments.RunResult.__module__ == "bettermdptools.experiments.types"
    assert experiments.OptunaResult.__module__ == "bettermdptools.experiments.optuna"
