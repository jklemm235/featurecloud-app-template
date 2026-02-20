# FeatureCloud Template

This repository is a starter template for building a [FeatureCloud app](https://featurecloud.ai/) that can be run in three ways:

- as a regular FeatureCloud app in true federated learning projects after publishing the app
- as a regular FeatureCloud app using FeatureClouds dockerized testembed, running on just one machine (no publishing needed)
- as a native local simulation (without Docker) just on one machine (no publishing needed)

## Where to add your federated learning algorithm

Implement your full federated learning logic in:

- `logic.py` → function `fl_algorithm(...)`

The template is already wired so that any execution via featurecloud or native execution
call this same function.

Relevant files:

- `logic.py`: All relevant federated learning logic should be in the `fl_algorithm(...)`! 
You can easily call helper classes such as a client or coordinator class from here and 
differentiate between client and coordinator via a simple `fed_learning_class_instance.is_coordinator` call.
- `helper/protocolfedlearningclass.py`: This protocol describes all the federated learning helper methods such as sending and receiving data you can run.
- `template_run_simulation.py`: This example script shows you how you can run a simulation with specific data.
- `requirements.txt`: Please make sure to add any dependencies you add to `logic.py`. 
Otherwise the dockerized execution of your app will fail, as the Dockerfile to dockerize your app uses this file!

## Dependencies

### 1) For native simulation
For native simulation of your federated learning app, you need to have all relevant python
packages of your algorithm installed. If you correctly added the dependencies to the `requirements.txt`, 
you can simply install them e.g. via `venv`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) For dockerized simulation
Just docker (docker engine or docker desktop)

## How to run simulations

The default simulation entry script is:

- `template_run_simulation.py`

It already contains both simulation calls:

- `run_simulation_featurecloud(...)`
- `run_simulation_native(...)`

Before running, adjust these values in `template_run_simulation.py` to your setup:

- `inputfolders`
- `outputfolders` (for native simulation)
- `generic_dir`
- `fl_algorithm_function`

### Run native simulation

Use the same script after setting your folders:

```bash
python template_run_simulation.py
```

If you only want native simulation, keep `run_simulation_native(...)` and comment out `run_simulation_featurecloud(...)`.

### Run Dockerized FeatureCloud simulation

Use the same script after setting your folders and making sure Docker is running:

```bash
python template_run_simulation.py
```

If you only want the Dockerized FeatureCloud test, keep `run_simulation_featurecloud(...)` and comment out `run_simulation_native(...)`.

## Publishing your app to Featurecloud
For others to use your app you need to publish it.
Please replace this `README.md` with a README that fits your actual app!
Then follow the steps provided [here](https://featurecloud.ai/developers)

## Project structure (important files)

- `logic.py`: your federated learning algorithm implementation. This is where you implement all app logic!
- `states.py`: minimal FeatureCloud state (`initial` → `terminal`). You should never need to touch this.
- `template_run_simulation.py`: simulation launcher template. This is just meant as inspiration to help understand how to write a small simulation script with some test data.
- `helper/run_app_simulation.py`: native + FeatureCloud simulation helpers. You should never need to touch this.
- `main.py`: FeatureCloud app server entrypoint. You should never need to touch this.
