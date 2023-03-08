import matplotlib.pyplot as plt


def read_file2(filename):
    f = open(filename, "r")
    split_caracter = "-"
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
            print('la ligne', i, 'na pas peu être chargé')

    return result


def plot(result, label1, label2, title=None):
    x = result[label1]
    y = result[label2]
    plt.plot(x, y)
    if not title is None:
        plt.title(title)
    plt.show()


def plot_cpu_vs_gpu(result):
    taille = result['taille']
    cpu = result['temps_cpu']
    gpu = result['temps_gpu']
    plt.plot(taille, cpu)
    plt.plot(taille, gpu)
    plt.legend(['temps CPU', 'temps GPU'])
    plt.show()


if __name__ == "__main__":
    result = read_file2("tailles.txt")
    # plot(result, 'taille', 'temps_gpu', 'temps du cpu en fonction de la taille')
    plot_cpu_vs_gpu(result)