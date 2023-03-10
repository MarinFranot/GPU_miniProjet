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


def plot(result, label1, label2, title=None, log=False, condition=None):
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
    
    plt.show()

def plot_gen(result, n1, n2):
    x = result[n1]
    y = result[n2]

    fig, ax = plt.subplots()
    ax.semilogx(x, y)
    ax.grid()
    plt.legend([n2])
    plt.show()
    
    

if __name__ == "__main__":
    result = read_file2("tailles.txt")
    #plot(result, 'taille', 'temps_cpu', title='temps du cpu en fonction de la taille', log=True)
    #plot(result, 'taille', 'temps_gpu', title='temps du gpu en fonction de la taille', log=True)
    #plot_cpu_vs_gpu(result)

    # result = read_file2("J.txt")
    # plot(result, 'J', 'temps_gpu', title='temps du gpu en fonction de J. arraysize = 10^6', condition={'taille':1e6})
    # plot(result, 'J', 'temps_gpu', title='temps du gpu en fonction de J. arraysize = 10^7', condition={'taille':1e7})

    # result = read_file2("K.txt")
    # plot(result, 'K', 'temps_gpu', title='temps du gpu en fonction de K. arraysize = 10^6', condition={'taille':1e6})
    # plot(result, 'K', 'temps_gpu', title='temps du gpu en fonction de K. arraysize = 10^7', condition={'taille':1e7})


    plot_gen(result, 'taille', 'computationThroughput(GOPS/s)\n')
    plot_gen(result, 'taille', 'memoryThroughput(GB/s)')
    

    


