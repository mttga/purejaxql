#!/bin/bash

gpu=0
seeds=(1) # (1 2 3 4 5)

# dm suite tasks
games=(
  AcrobotSwingup
  AcrobotSwingupSparse
  BallInCup
  CartpoleBalance
  CartpoleBalanceSparse
  CartpoleSwingup
  CartpoleSwingupSparse
  CheetahRun
  FingerSpin
  FingerTurnEasy
  FingerTurnHard
  FishSwim
  HopperHop
  HopperStand
  HumanoidStand
  HumanoidWalk
  HumanoidRun
  PendulumSwingup
  PointMass
  ReacherEasy
  ReacherHard
  SwimmerSwimmer6
  WalkerRun
  WalkerStand
  WalkerWalk
)

for game in "${games[@]}"
do
  for seed in "${seeds[@]}"
  do
    job_name="${game}_${seed}"

    docker run -it --rm --gpus="device=${gpu}" --ipc=host \
        -v $(pwd):/app \
        -w /app \
        --name "pqn-${gpu//,/-}" \
        pqn \
        python purejaxql/pqn_mujoco_playground.py \
        +alg=pqn_playground_dm_suite alg.ENV_NAME=${game} SEED=${seed}
  done
done



#  locomotion tasks
games=(
  BerkeleyHumanoidJoystickFlatTerrain
  BerkeleyHumanoidJoystickRoughTerrain
  G1JoystickFlatTerrain
  G1JoystickRoughTerrain
  Go1Footstand
  Go1Getup
  Go1Handstand
  Go1JoystickFlatTerrain
  Go1JoystickRoughTerrain
  H1InplaceGaitTracking
  H1JoystickGaitTracking
  SpotFlatTerrainJoystick
  SpotGetup
  SpotJoystickGaitTracking
  T1JoystickFlatTerrain
)


for game in "${games[@]}"
do
  for seed in "${seeds[@]}"
  do
    job_name="${game}_${seed}"

    docker run -it --rm --gpus="device=${gpu}" --ipc=host \
        -v $(pwd):/app \
        -w /app \
        --name "pqn-${gpu//,/-}" \
        pqn \
        python purejaxql/pqn_mujoco_playground.py \
        +alg=pqn_playground_locomotion alg.ENV_NAME=${game} SEED=${seed}

  done
done


# locomotion tasks with gamma=0.97
games=(
  ApolloJoystickFlatTerrain
  BarkourJoystick
  Op3Joystick
  T1JoystickRoughTerrain
)

for game in "${games[@]}"
do
  for seed in "${seeds[@]}"
  do
    job_name="${game}_${seed}"

    docker run -it --rm --gpus="device=${gpu}" --ipc=host \
        -v $(pwd):/app \
        -w /app \
        --name "pqn-${gpu//,/-}" \
        pqn \
        python purejaxql/pqn_mujoco_playground.py \
        +alg=pqn_playground_locomotion alg.ENV_NAME=${game} SEED=${seed} alg.GAMMA=0.97

  done
done


# manipulation tasks
games=(
  AlohaHandOver
  AlohaSinglePegInsertion
  LeapCubeRotateZAxis
  LeapCubeReorient
  PandaPickCubeCartesian
  PandaRobotiqPushCube
)

for game in "${games[@]}"
do
  for seed in "${seeds[@]}"
  do
    job_name="${game}_${seed}"

    docker run -it --rm --gpus="device=${gpu}" --ipc=host \
        -v $(pwd):/app \
        -w /app \
        --name "pqn-${gpu//,/-}" \
        pqn \
        python purejaxql/pqn_mujoco_playground.py \
        +alg=pqn_playground_manipulation alg.ENV_NAME=${game} SEED=${seed}


  done
done

games=(
  PandaPickCube
  PandaPickCubeOrientation
)

for game in "${games[@]}"
do
  for seed in "${seeds[@]}"
  do
    job_name="${game}_${seed}"

    docker run -it --rm --gpus="device=${gpu}" --ipc=host \
        -v $(pwd):/app \
        -w /app \
        --name "pqn-${gpu//,/-}" \
        pqn \
        python purejaxql/pqn_mujoco_playground.py \
        +alg=pqn_playground_manipulation alg.ENV_NAME=${game} SEED=${seed} alg.GAMMA=0.95 alg.LAMBDA=0.5

  done
done