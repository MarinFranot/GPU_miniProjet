import matplotlib.pyplot as plt


def read_file2(filename):
    f = open(filename, "r")
    split_caracter = "|"
    result = {}
    labels = f.readline()[:-1].split(split_caracter)
    for label in labels:
        result[label] = []
    print('labels :', labels)
    print('info :', f.readline())

    for i, x in enumerate(f):
        line = x[:-1].split(split_caracter)
        try:
            for j, label in enumerate(labels):
                result[label].append(float(line[j]))
        except:
            print('la ligne', i + 3, 'na pas peu être chargé')
            print(x)

    return result


def plot(result, label1, label2, title=None, log=False, condition=None, axis_label=None):
    """ 
    Affiche label1 en fonction de label2 d'un résultat: result, avec le titre: title, une possibilitée d'avoir une échelle log avec log=True
    result est un dictionnaire de tableau (de même tailles)
    on peut aussi mettre une contition sous la forme: condition = {'label3': 1e6}
    """
    if condition is None:
        x = result[label1]
        y = result[label2]
    else:
        x , y = [], []
        for i in range(len(result[label1])):
            add = True
            for key, value in condition.items():
                if result[key][i] != value:
                    add = False
            if add:
                x.append(result[label1][i])
                y.append(result[label2][i])


    if not log:
        plt.figure(figsize=(8,6))
        plt.plot(x, y)
        if not axis_label is None:
            plt.xlabel(axis_label[0])
            plt.ylabel(axis_label[1])
        plt.grid()
        

    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.semilogx(x, y)
        if not axis_label is None:
            ax.set_xlabel(axis_label[0])
            ax.set_ylabel(axis_label[1])
        ax.grid()

    if not title is None:
            plt.title(title)

    plt.show()


def plot_cpu_vs_gpu(result):
    taille = result['taille']
    cpu = result['temps_cpu']
    gpu = result['temps_gpu']

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogx(taille, cpu)
    ax.semilogx(taille, gpu)
    ax.grid()
    plt.legend(['CPU computation time', 'GPU computation time'])
    plt.title('comparison between CPU and GPU computation time as a function of array size')
    axis_label = ['arraysize', 'computation time (ms)']
    ax.set_xlabel(axis_label[0])
    ax.set_ylabel(axis_label[1])
    plt.show()


def plot_roofline(result, condition=None):
    label1 = 'memoryThroughput(GB/s)'
    label2 = 'computationThroughput(GOPS/s)'
    compute_intensity = []
    computation_throughput = []

    if condition is None:
        compute_intensity = [result[label1][i]/result[label2][i] for i in range(len(result[label1]))]
        computation_throughput = result[label2]
    else:
        compute_intensity = []
        computation_throughput = []
        for i in range(len(result[label1])):
            add = True
            for key, value in condition.items():
                if result[key][i] != value:
                    add = False
            if add:
                compute_intensity.append(result[label1][i]/result[label2][i])
                computation_throughput.append(result[label2][i])
    print(f'{compute_intensity = }')
    print(f'{computation_throughput = }')
    plt.plot(compute_intensity, computation_throughput, '+')
    plt.show()



if __name__ == "__main__":
    # result = read_file2("tailles.txt")
    # axis_label = ['arraysize', 'computation time (ms)']
    # plot(result, 'taille', 'temps_cpu', title='CPU computation time as a function of vector size', log=True, axis_label=axis_label)
    # plot(result, 'taille', 'temps_gpu', title='GPU computation time as a function of array size', log=True, axis_label=axis_label)
    # plot_cpu_vs_gpu(result)

    # result = read_file2("J.txt")
    # axis_label = ['J', 'computation time (ms)']
    # plot(result, 'J', 'temps_gpu', title='GPU computation time as a function of J (arraysize = 10^6)', condition={'taille':1e6}, axis_label=axis_label)
    # plot(result, 'J', 'temps_gpu', title='GPU computation time as a function of J (arraysize = 10^7)', condition={'taille':1e7}, axis_label=axis_label)

    result = read_file2("K.txt")
    # axis_label = ['K', 'computation time (ms)']
    # plot(result, 'K', 'temps_gpu', title='GPU computation time as a function of K (arraysize = 10^6)', condition={'taille':1e6}, axis_label=axis_label)
    # plot(result, 'K', 'temps_gpu', title='GPU computation time as a function of K (arraysize = 10^7)', condition={'taille':1e7}, axis_label=axis_label)
    plot_roofline(result, condition={'taille':1e6})
