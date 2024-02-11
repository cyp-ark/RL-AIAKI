def train(params, rng, loader_train, loader_validation):
    qnet = RL(state_dim=params["embed_state_dim"], nb_actions=params["num_actions"], gamma=params["gamma"], learning_rate=params["rl_learning_rate"], 
                update_freq=params["update_freq"], rng, device=params["device"])
    


    expt = DQNExperiment(data_loader_train=loader_train, data_loader_validation=loader_validation, q_network=qnet, ps=0, ns=2,
                        folder_location=params["folder_location"], folder_name=params["folder_name"], 
                        saving_period=params["exp_saving_period"], rng=rng, resume=params["rl_resume"])
                        



