from multiprocessing import Process
def showSquare(num=2):
    for i in range(1000000):
        pass
    print(num**2)
procs = []
for i in range(5):
    procs.append(Process(target=showSquare))
for proc in procs:
    proc.start()
print("Hello")
for proc in procs:
    proc.join()