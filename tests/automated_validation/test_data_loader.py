from vivarium_testing_utils.automated_validation.data_loader import DataManager


def test_load_from_simulation(sim_result_dir):
    data_loader = DataManager(sim_result_dir)
    person_time_cause = data_loader.load_from_sim("person_time_cause")
    assert person_time_cause.shape == (100, 3)
    assert person_time_cause.columns.tolist() == ["age", "sex", "value"]


def test_sim_outputs(sim_result_dir):
    data_loader = DataManager(sim_result_dir)
    assert data_loader.sim_outputs() == [
        "person_time_cause",
        "transition_count_cause",
        "deaths",
        "ylls",
        "ylds",
    ]
