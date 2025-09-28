for config_file in configs/*.yaml; do
    uv run run-sim run "$config_file"
done
