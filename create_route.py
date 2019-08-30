import random
import sys

if __name__ == "__main__":
    rand_seed = int(sys.argv[1])

    random.seed(rand_seed)  # make tests reproducible
    N = 7200  # number of time steps

    rand_num = random.uniform(0, 1)
    probability = 1.   # for initialization

    if(rand_num <= 0.33):
        print('low traffic')
        probability = 0.1
    elif(rand_num <= 0.66):
        print('mid traffic')
        probability = 0.3
    else:
        probability = 0.5
        print('high traffic')

    with open("cross.rou.xml", "w") as routes:
        print("""<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <route edges="-gneE16 gneE17" color="yellow" id="route_0"/>
    <route edges="-gneE16 gneE13" color="yellow" id="route_1"/>
    <route edges="-gneE16 gneE13" color="magenta" id="route_10"/>
    <route edges="-gneE16 gneE15" color="magenta" id="route_11"/>
    <route edges="-gneE13 gneE17" color="yellow" id="route_12"/>
    <route edges="-gneE13 gneE16" color="yellow" id="route_13"/>
    <route edges="-gneE13 gneE15" color="yellow" id="route_14"/>
    <route edges="-gneE16 gneE15" color="yellow" id="route_2"/>
    <route edges="-gneE15 gneE16" color="red" id="route_3"/>
    <route edges="-gneE15 gneE17" color="red" id="route_4"/>
    <route edges="-gneE15 gneE13" color="red" id="route_5"/>
    <route edges="-gneE17 gneE13" color="blue" id="route_6"/>
    <route edges="-gneE17 gneE15" color="blue" id="route_7"/>
    <route edges="-gneE17 gneE16" color="blue" id="route_8"/>
    <route edges="-gneE16 gneE17" color="magenta" id="route_9"/>
""", file=routes)

        vehNr = 0

        for i in range(N):
            for route_index in range(15):
                if random.uniform(0, 1) < probability:
                    print(i, ' + ', route_index)
                    print('    <vehicle id="%i" route="route_%i" depart="%i" />' % (
                        vehNr, route_index, i), file=routes)
                    vehNr += 1
        print("</routes>", file=routes)

