# Model Predictive Control

This repository was created for the purposes of SC42154 - Model Predictive Control. 

## Install dependecies

Before running the script make sure to install all dependecies by running

```bash
pip install -r requirements.txt
```

## How to run

In order to run the script first a trajectory needs to be generated this can be done by running the following command
```bash
python3 quadrotor.py
```

A menu will pop up where you can select reference trajectory and initial state for the quadrotor. 
The first pop up menu will ask you which trajectory you will like the drone to follow. The options are:

| Option Name       | Description                                  |
|-------------------|----------------------------------------------|
|     Pre loaded   | Uses previously generated trajectory (If this is the first time running the script a trajectory does not yet exist)              |
| Single Point        | Allow you to select a single point and the quadrotor will go to it|
|Circle | Allow you to create a circle of variable height and radius and in the XY plane          |
| Tu Delft    | Created the TU Delft Logo    |
|Bread | Creates a bread shape|

After the trajectory type has been selected you can choose the initial state of the quadrotor and whether you want the system to have disturbances or not. Finally you will be asked if you wish for the script to be run in debug mode.

Note: The trajectories like TU Delft and Bread take a long time to run especially in the case with disturbances where the OTS has to be solved online. 
Note: Debug is only meant for debugging the script by printing additional information to the console.