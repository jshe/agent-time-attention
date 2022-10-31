bash run_many.sh python3 main.py --config="coma"
                                      --env-config="dk" with mac="basic_mac" \
                                                                env_args.p=1.0,1.0 \
                                                                arch="lstm" \
                                                                rr="rudder" \
                                                                local_results_path="results/mr/rudder"

bash run_many.sh python3 main.py --config="dk"
                                      --env-config="mr" with mac="basic_mac" \
                                                                env_args.p=1.0,1.0  \
                                                                arch="none" \
                                                                rr="none" \
                                                                local_results_path="results/mr/none"