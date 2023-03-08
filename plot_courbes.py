import matplotlib.pyplot as plt


def read_file2(filename):
    f = open(filename, "r")
    split_caracter = "|"
    result = {}
    labels = f.readline().split(split_caracter)
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


def plot(result, label1, label2, title=None, log=False):
    x = result[label1]
    y = result[label2]

    if not log:
        plt.plot(x, y)

    else:
        fig, ax = plt.subplots()
        ax.semilogx(x, y)
        ax.grid()

    if not title is None:
            plt.title(title)

    plt.show()


def plot_cpu_vs_gpu(result):
    taille = result['taille']
    cpu = result['temps_cpu']
    gpu = result['temps_gpu']

    fig, ax = plt.subplots()
    ax.semilogx(taille, cpu)
    ax.semilogx(taille, gpu)
    ax.grid()
    plt.legend(['temps CPU', 'temps GPU'])
    plt.show()


if __name__ == "__main__":
    result = read_file2("tailles.txt")
    plot(result, 'taille', 'temps_cpu', 'temps du cpu en fonction de la taille', log=True)
    plot(result, 'taille', 'temps_gpu', 'temps du gpu en fonction de la taille', log=True)
    plot_cpu_vs_gpu(result)
