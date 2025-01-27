if __name__ == "__main__":
    from helpers.data_loading import load_data
    from helpers.data_processing import preprocess_data
    from helpers.multiplier_fitting import fit_multipliers
    from helpers.data_merging import merge_data
    from helpers.weight_assignment import assign_weights
    from helpers.simulation import simulate_lift_data
    from helpers.functional_max_calc import calculate_functional_maxes

    # Step 1: Load Data
    program_df, maxes_df = load_data()

    # Step 2: Preprocess Data
    flattened_core_maxes_df, repeated_program_df = preprocess_data(program_df, maxes_df)

    # Step 3: Fit Multipliers
    exercise_functions = fit_multipliers(repeated_program_df)  # ✅ Now safe for multiprocessing

    # Step 4: Merge Data
    merged_data = merge_data(program_df, maxes_df)

    # Step 5: Calculate Assigned Weights
    assigned_weights_df = assign_weights(merged_data, flattened_core_maxes_df, exercise_functions)

    # Step 6: Simulate Lifting Performance
    simulated_data = simulate_lift_data(assigned_weights_df, num_iterations=5)

    # Step 7: Calculate Functional Maxes
    functional_maxes = calculate_functional_maxes(simulated_data)
