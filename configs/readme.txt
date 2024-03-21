
{
  "experiment_name": "debug_moo_pam_small",
  "checkpoints_dir": "./checkpoints",
  "continue_train":false,
  "debug":false,

  "model": {
    "name": "PAM",
    "nepoch":  1,
    "batch_size":2,
    "init_training_size":20,
    "eval_freq":1
  },
  "model1": {
    "name": "AD",
    "nepoch":  20,
    "batch_size":1,
    "init_training_size":20,
    "eval_freq":1,
    "nbootstrap":5,
    "selector":"KG",
    "x_i":0.01
  },
  "dataset":{
    "name": "MOO",
    "fix_init_tr":true,
    "yield_target":0.8,
    "simulator_so":{
      "type": "ml_learner",
      "name": "XGBoost",
      "ds_name":"cqd_raw"
    },
    "simulator_moo":{
      "f_color":{
        "type": "functions",
        "name": "cubic",
	      "title":"cubic_color",
        "overwrite":false,
        "global_space_rng_pth": "./data/simulator/fake/moo_global_small.json"
      },
      "f_yield":{
        "type": "functions",
        "name": "linear",
	      "title":"linear_yield",
        "overwrite":false,
        "global_space_rng_pth": "./data/simulator/fake/moo_global_small.json"
      }

    },
    "real":{
      "data_pth": "./data/xxx.csv",
      "global_space_rng_pth": "./data/simulator/ml_learners/cqd_raw_global_space.json"
    }
  }

  
}
