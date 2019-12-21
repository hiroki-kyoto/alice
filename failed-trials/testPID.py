import pid as PID
import matplotlib.pyplot as plt

# configuration for simulation
STEPS = 50
INERTIA_TIME = 3
SAMPLE_TIME = 0.3

plt.figure(1)


def TestPID(P, I, D):
    IncrementalPid = PID.IncrementalPID(P, I, D)
    PositionalPid = PID.PositionalPID(P, I, D)
    IncrementalXaxis = [0]
    IncrementalYaxis = [0]
    PositionalXaxis = [0]
    PositionalYaxis = [0]

    for i in range(1, STEPS):
        IncrementalPid.SetStepSignal(100.2)
        IncrementalPid.SetInertiaTime(INERTIA_TIME, SAMPLE_TIME)
        IncrementalYaxis.append(IncrementalPid.SystemOutput)
        IncrementalXaxis.append(i)

        PositionalPid.SetStepSignal(100.2)
        PositionalPid.SetInertiaTime(INERTIA_TIME, SAMPLE_TIME)
        PositionalYaxis.append(PositionalPid.SystemOutput)
        PositionalXaxis.append(i)

    plt.figure(1)
    plt.plot(range(0, STEPS), [100.2]*STEPS, '--')
    plt.plot(IncrementalXaxis, IncrementalYaxis, 'r+')
    plt.plot(PositionalXaxis, PositionalYaxis, 'b*')
    plt.xlim(0, STEPS)
    plt.ylim(0, 140)
    plt.legend(['Real', 'Incremental PID Control', 'Positional PID Control'])

    plt.show()


if __name__ == "__main__":
    TestPID(4.5, 0.5, 0.1)
