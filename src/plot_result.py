import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import os
import sys

def plotQoE(network, movie):
    # ABR理论性能
    path_QoE = '../performance/' + network + '_' + movie
    file_list = os.listdir(path_QoE)
    file_count = len(file_list)

    abr_name = []
    time_average_played_bitrate = np.zeros(file_count)
    time_average_rebuffer = np.zeros(file_count)
    time_average_bitrate_change = np.zeros(file_count)
    bandwidth_utilization = np.zeros(file_count)
    quality_of_experience = np.zeros(file_count)

    for i in range(file_count):
        file = open(path_QoE + '/' + file_list[i], 'r')
        abr_name.append(file.readline().replace("\n", "").replace("ABR name: ", ""))
        file.readline()
        file.readline()
        file.readline()
        file.readline()
        time_average_played_bitrate[i] = float(file.readline().replace("\n", "").replace("time average played bitrate: ", ""))
        file.readline()
        file.readline()
        file.readline()
        file.readline()
        time_average_rebuffer[i] = float(file.readline().replace("\n", "").replace("time average rebuffer: ", ""))
        file.readline()
        file.readline()
        file.readline()
        time_average_bitrate_change[i] = float(file.readline().replace("\n", "").replace("time average bitrate change: ", ""))
        file.readline()
        file.readline()
        file.readline()
        file.readline()
        file.readline()
        file.readline()
        file.readline()
        file.readline()
        file.readline()
        file.readline()
        file.readline()
        file.readline()
        bandwidth_utilization[i] = float(file.readline().replace("\n", "").replace("bandwidth utilization: ", ""))*100
        quality_of_experience[i] = float(file.readline().replace("\n", "").replace("Quality of Experience: ", ""))
        file.close()

    plt.figure('bitrate')
    plt.bar(abr_name, time_average_played_bitrate)
    for a, b in zip(abr_name, time_average_played_bitrate):
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=10)
    plt.title('time average bitrate (kbps)')
    plt.savefig('../figures/' + network + '_' + movie + '/performance_bitrate.png')

    plt.figure('rebuffer')
    plt.bar(abr_name, time_average_rebuffer)
    for a, b in zip(abr_name, time_average_rebuffer):
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=10)
    plt.title('time average rebuffer')
    plt.savefig('../figures/' + network + '_' + movie + '/performance_rebuffer.png')

    plt.figure('bitrate change')
    plt.bar(abr_name, time_average_bitrate_change)
    for a, b in zip(abr_name, time_average_bitrate_change):
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=10)
    plt.title('time average bitrate change (kbps)')
    plt.savefig('../figures/' + network + '_' + movie + '/performance_bitrate_change.png')

    plt.figure('bandwidth utilization')
    plt.bar(abr_name, bandwidth_utilization)
    for a, b in zip(abr_name, bandwidth_utilization):
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=10)
    plt.title('bandwidth utilization (%)')
    plt.savefig('../figures/' + network + '_' + movie + '/performance_bandwidth_utilization.png')

    plt.figure('QoE')
    plt.bar(abr_name, quality_of_experience)
    for a, b in zip(abr_name, quality_of_experience):
        plt.text(a, b, '%.0f' % b, ha='center', va='bottom', fontsize=10)
    plt.title('quality of experience')
    plt.savefig('../figures/' + network + '_' + movie + '/performance_QoE.png')

    # plt.show()


def plotChunkRecord(network, movie):
    # 视频块大小记录
    path_chunk = '../chunk_log/' + network + '_' + movie
    file_list = os.listdir(path_chunk)
    file_count = len(file_list)
    chunk_record = np.zeros((file_count, 199))
    bandwidth_record = np.zeros((file_count, 199))
    x = list(range(1, 200))

    for i in range(file_count):
        file = open(path_chunk + '/' + file_list[i], 'r')
        for j in range(199):
            tmp = file.readline().replace('\n', '').split('/')
            chunk_record[i, j] = float(tmp[0])
            bandwidth_record[i, j] = float(tmp[1])

    for i in range(file_count):
        plt.figure(i)
        plt.title('Bitrate variation')
        l1, = plt.plot(x, chunk_record[i], marker='o', markersize=3)
        l2, = plt.plot(x, bandwidth_record[i], marker='o', markersize=3)
        # print(file_list[i])
        file_name = file_list[i].replace('.txt', '')
        # print(file_name)
        plt.legend(handles=[l1, l2, ], labels=[file_name, 'bandwidth'])
        if movie == 'bbb4k':
            plt.axhline(y=1000, ls='--')
            plt.axhline(y=2500, ls='--')
            plt.axhline(y=5000, ls='--')
            plt.axhline(y=8000, ls='--')
            plt.axhline(y=16000, ls='--')
            plt.axhline(y=35000, ls='--')
        elif movie == 'bbb':
            plt.axhline(y=230, ls='--')
            plt.axhline(y=331, ls='--')
            plt.axhline(y=477, ls='--')
            plt.axhline(y=688, ls='--')
            plt.axhline(y=991, ls='--')
            plt.axhline(y=1427, ls='--')
            plt.axhline(y=2056, ls='--')
            plt.axhline(y=2962, ls='--')
            plt.axhline(y=5027, ls='--')
            plt.axhline(y=6000, ls='--')
        plt.savefig('../figures/' + network + '_' + movie + '/chunk_log_' + file_name + '.png')
    # plt.show()


if __name__ == '__main__':
    network = sys.argv[1]
    movie = sys.argv[2]
    os.makedirs('../figures/' + network + '_' + movie, exist_ok=True)
    plotQoE(network, movie)
    plotChunkRecord(network, movie)
