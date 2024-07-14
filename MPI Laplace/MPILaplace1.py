from mpi4py import MPI
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# initializing variables
ROWS, COLUMNS = 1000, 1000
MAX_TEMP_ERROR = 0.01

# MPI related variavles
rank = MPI.COMM_WORLD.Get_rank()
numPE = MPI.COMM_WORLD.Get_size()
startTime = MPI.Wtime()


def initialize_temperature(temp):
    temp[:, :] = 0

    # Set right side boundery condition
    for i in range(ROWS + 1):
        temp[i, COLUMNS + 1] = (100 / ROWS) * i
    # Set bottom boundery condition
    for i in range(COLUMNS + 1):
        temp[ROWS + 1, i] = (100 / ROWS) * i
    return temp


temperature_last = np.empty((ROWS + 2, COLUMNS + 2))
initialize_temperature(temperature_last)

# braodcast input
max_iterations = 0
if rank == 0:
    max_iterations = int(input("Maximum iterations: "))

max_iterations = MPI.COMM_WORLD.bcast(max_iterations, root=0)  # send the max interations to all PE

# define each PE's job
PErow = ROWS // 4
startRow = rank * PErow + 1
endRow = startRow + PErow - 1

# Adjust for ghost rows
# PE 0 has ghost row below; PE3 has ghost row above
# PE 1 and 2 has ghost row both above and below
if rank != 0:
    startRow -= 1
if rank != numPE - 1:
    endRow += 1

curTemp = np.empty((PErow + 1, COLUMNS + 1))  # define curTemp for each PE
dt = 100
iteration = 1
count = 0
while (dt > MAX_TEMP_ERROR) and (iteration < max_iterations):
    # computing the number
    for i in range(1, PErow + 1):
        for j in range(1, COLUMNS + 1):
            # updating the current Temeperature that each PE is dealing with the correspinding data from temperature_last
            curTemp[i, j] = 0.25 * (temperature_last[startRow + i + 1, j] + temperature_last[startRow + i - 1, j] +
                                    temperature_last[startRow + i, j + 1] + temperature_last[startRow + i, j - 1])
    MPI.COMM_WORLD.barrier()

    # data passing logics
    if rank != 0:
        MPI.COMM_WORLD.Send(curTemp[1, :], dest=rank - 1, tag=0)  # send the second row up
        # print(f"PE {rank} sent to {rank-1}", flush=True)
        print(flush=True)
    if rank != 3:
        MPI.COMM_WORLD.Send(curTemp[-2, :], dest=rank + 1, tag=1)  # send the second to last row down
        # print(f"PE {rank} sent to {rank+1}", flush=True)
        print(flush=True)
    if rank != 0:
        MPI.COMM_WORLD.Recv(curTemp[0, :], source=rank - 1, tag=1)  # recv from up, sotre in the first row
        # print(f"PE {rank} recieved from PE {rank-1}", flush=True)
        print(flush=True)

    if rank != 3:
        MPI.COMM_WORLD.Recv(curTemp[-1, :], source=rank + 1, tag=0)  # recv from down, store in the last row
        # print(f"PE {rank} recieved from PE {rank+1}", flush=True)
        print(flush=True)

    # compute dt
    dt = 0
    for i in range(1, PErow + 1):
        for j in range(1, COLUMNS + 1):
            dt = max(dt, curTemp[i, j] - temperature_last[startRow + i, j])
            temperature_last[startRow + i, j] = curTemp[i, j]

    # collect all dt, and compute the max, and then bcast them to each PE
    MPI.COMM_WORLD.reduce(dt, op=MPI.MAX, root=0)
    dt = MPI.COMM_WORLD.bcast(dt, root=0)

    print("dt: ", dt)
    print("iteration: %d" % iteration)
    iteration += 1
    # save the total time
    endTime = MPI.Wtime()
    totaltime = endTime - startTime
    print(totaltime)
    if rank == 0:
        with open("MPI_total_time.txt", "w") as file:
            file.write(str(totaltime))

# print(curTemp.shape)
# print(curTemp)

# plot the 4 pieces of curTemp
if rank == 0:
    plt.imshow(curTemp, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Temperature')
    plt.title("PE0Temperature Distribution")
    plt.savefig("full_plate0.png")
    plt.show()
    # Visualize the bottom corner
    corner_data = curTemp[-5:, -5:]
    plt.imshow(corner_data, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Temperature')
    plt.title("PE0Zoomed Bottom Corner")
    plt.savefig("bottom_corner0.png")
    plt.show()
elif rank == 1:
    plt.imshow(curTemp, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Temperature')
    plt.title("PE1Temperature Distribution")
    plt.savefig("full_plate1.png")
    plt.show()
    # Visualize the bottom corner
    corner_data = curTemp[-5:, -5:]
    plt.imshow(corner_data, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Temperature')
    plt.title("PE1Zoomed Bottom Corner")
    plt.savefig("bottom_corner1.png")
    plt.show()
elif rank == 2:
    plt.imshow(curTemp, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Temperature')
    plt.title("PE2Temperature Distribution")
    plt.savefig("full_plate2.png")
    plt.show()
    # Visualize the bottom corner
    corner_data = curTemp[-5:, -5:]
    plt.imshow(corner_data, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Temperature')
    plt.title("PE2Zoomed Bottom Corner")
    plt.savefig("bottom_corner2.png")
    plt.show()
elif rank == 3:
    plt.imshow(curTemp, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Temperature')
    plt.title("PE3Temperature Distribution")
    plt.savefig("full_plate3.png")
    plt.show()
    # Visualize the bottom corner
    corner_data = curTemp[-5:, -5:]
    plt.imshow(corner_data, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Temperature')
    plt.title("PE3Zoomed Bottom Corner")
    plt.savefig("bottom_corner3.png")
    plt.show()

