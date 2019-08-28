import traci
from sumolib import checkBinary

sumoBinary = checkBinary('sumo-gui')
sumoCmd = [sumoBinary, "-c", "sumo/cross.sumocfg"]

traci.start(sumoCmd)
step = 0
while step < 1000:
    traci.simulationStep()
    if traci.inductionloop.getLastStepVehicleNumbe("0") > 0:
        traci.trafficlight.setRedYellowGreenState("0", "GrGr")
    step += 1

traci.close()
