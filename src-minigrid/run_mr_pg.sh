bash run_many.sh python3 main.py --config="pg" \
                                      --env-config="mr" with mac="distributed_mac" \
                                                                env_args.p=1.0,1.0 \
                                                                arch="lstm" \
                                                                rr="rudder" \
                                                                local_results_path="results/mr/rudder"

bash run_many.sh python3 main.py --config="pg" \
                                      --env-config="mr" with mac="distributed_mac" \
                                                                env_args.p=1.0,1.0 \
                                                                arch="transformer" \
                                                                rr="ata" \
                                                                local_results_path="results/mr/ata"

bash run_many.sh python3 main.py --config="pg" \
                                      --env-config="mr" with mac="distributed_mac" \
                                                                env_args.p=1.0,1.0 \
                                                                arch="none" \
                                                                rr="none" \
                                                                local_results_path="results/mr/none"