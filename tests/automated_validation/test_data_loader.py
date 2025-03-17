from vivarium_testing_utils.automated_validation.data_loader import DataManager


def test_sim_outputs(sim_result_dir):
    data_loader = DataManager(sim_result_dir)
    assert data_loader.sim_outputs() == [
        "deaths",
        "person_time_cause",
        "transition_count_cause",
    ]


def test_load_from_simulation(sim_result_dir):
    data_loader = DataManager(sim_result_dir)
    person_time_cause = data_loader.load_from_sim("deaths")
    assert person_time_cause.shape == (8, 1)
    # check that value is column and rest are indices
    assert person_time_cause.index.names == [
        "measure",
        "entity_type",
        "entity",
        "sub_entity",
        "age_group",
        "sex",
        "input_draw",
        "random_seed",
    ]
    assert person_time_cause.columns == ["value"]
