import bettermdptools.experiments as experiments
import bettermdptools.plotting as plotting


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


def test_plotting_exports_preparation_renderers_and_data_types():
    assert set(plotting.__all__) == {
        "ConvergenceAxes",
        "ConvergenceData",
        "LearningCurveData",
        "PolicyGridData",
        "ValueGridData",
        "ValuePolicyAxes",
        "aggregate_values",
        "plot_convergence",
        "plot_learning_curve",
        "plot_policy_convergence",
        "plot_policy_grid",
        "plot_value_convergence",
        "plot_value_heatmap",
        "plot_value_policy",
        "prepare_convergence",
        "prepare_learning_curve",
        "prepare_policy_grid",
        "prepare_value_grid",
    }
