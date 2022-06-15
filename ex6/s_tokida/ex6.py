import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from io import BytesIO
from PIL import Image

def pca(data):
    """principal component analysis
    Args:
        data (pandas.core.frame.DataFrame): input data
    Returns:
        eigen_sorted (ndarray): eigenvalue and eigenvector (sorted)
        c_rate (ndarray): contribution rate
    """

    cov_matrix = np.cov(data.T)
    eigen_val, eigen_vec = np.linalg.eig(cov_matrix)
    eigen = np.hstack([eigen_val[:, np.newaxis], eigen_vec.T])
    # sort by eigen_val
    eigen_sorted = np.array(sorted(eigen, key=lambda x: x[0], reverse=True))
    eigen_vec_sorted = eigen_sorted[:, 1:].T
    # calculate contribution rate
    c_rate = eigen_sorted[:, 0] / sum(eigen_sorted[:, 0])

    return eigen_vec_sorted, c_rate


def render_frame(data, eigen, c_rate, angle, filename):
    """define render frame to make gif
    Args:
        df (ndarray): input data
        angle (int): angle
        filename (str): filename
    Returns:
        gif images
    """

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c = 'blueviolet', label = 'Observed data')
    x1 = np.linspace(min(data[:, 0]), max(data[:, 0]), data.shape[0])
    eigen_x2 = eigen[1] / eigen[0]
    eigen_x3 = eigen[2] / eigen[0]
    ax.plot(x1, eigen_x2[0]*x1, eigen_x3[0]*x1, color = 'c', label = f'c rate={c_rate[0]:.3f}')
    ax.plot(x1, eigen_x2[1]*x1, eigen_x3[1]*x1, color = 'y', label = f'c rate={c_rate[1]:.3f}')
    ax.plot(x1, eigen_x2[2]*x1, eigen_x3[2]*x1, color = 'm', label = f'c rate={c_rate[2]:.3f}')

    ax.view_init(30, angle)
    plt.close()

    ax.set(xlabel = "$x_1$", ylabel = "$x_2$", zlabel = "$x_3$", title = f'{filename}_pca')
    ax.legend()

    buf = BytesIO()
    fig.savefig(buf, bbox_inches = 'tight', pad_inches=0.0)
    return Image.open(buf)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('filepath', type=str, help='csv file name')
    args = parser.parse_args()

    filepath = args.filepath
    filename = filepath.split('.')[0]

    data = pd.read_csv(f'../{filepath}', header = None)

    # normalize data
    data_norm = data.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    eigen, c_rate = pca(data_norm)

    # data = data.values
    data = data_norm.values

    if data.shape[1] == 2:

        fig, ax = plt.subplots(figsize = (7,6))
        ax.set(xlabel = "$x_1$", ylabel = "$x_2$", title = f'{filename}')
        ax.scatter(data[:, 0], data[:, 1], c = 'blueviolet', label = 'Observed data')
        x1 = np.linspace(min(data[:, 0]), max(data[:, 0]), data.shape[0])
        
        # vec_s = [0, 0]
        # ax.quiver(vec_s[0], vec_s[1], eigen[0][1], eigen[0][2], angles='xy', scale_units='xy', scale=1, color = 'orange')
        # ax.quiver(vec_s[0], vec_s[1], eigen[1][1], eigen[1][2], angles='xy', scale_units='xy', scale=1, color = 'pink')
        
        eigen_x2 = eigen[1] / eigen[0]
        ax.plot(x1, eigen_x2[0]*x1, color = 'c', label = f'c rate={c_rate[0]:.3f}')
        ax.plot(x1, eigen_x2[1]*x1, color = 'y', label = f'c rate={c_rate[1]:.3f}')
        
        plt.legend()
        plt.tight_layout()
        # plt.savefig('fig/' + f'{filename}_pca.png')
        plt.show()
        plt.close()

    elif data.shape[1] == 3:

        fig = plt.figure(figsize = (7,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set(xlabel = "$x_1$", ylabel = "$x_2$", zlabel = "$x_3$", title = f'{filename}')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c = 'blueviolet', label = 'Observed data')
        x1 = np.linspace(min(data[:, 0]), max(data[:, 0]), data.shape[0])
        eigen_x2 = eigen[1] / eigen[0]
        eigen_x3 = eigen[2] / eigen[0]
        ax.plot(x1, eigen_x2[0]*x1, eigen_x3[0]*x1, color = 'c', label = f'c rate={c_rate[0]:.3f}')
        ax.plot(x1, eigen_x2[1]*x1, eigen_x3[1]*x1, color = 'y', label = f'c rate={c_rate[1]:.3f}')
        ax.plot(x1, eigen_x2[2]*x1, eigen_x3[2]*x1, color = 'm', label = f'c rate={c_rate[2]:.3f}')

        plt.legend()
        plt.tight_layout()
        # plt.savefig('fig/' + f'{filename}_pca.png')
        plt.show()
        plt.close()

        # gif image
        # images = [render_frame(data, eigen, c_rate, angle*2, filename) for angle in range(90)]
        # images[0].save('fig/' + f'{filename}_pca.gif', save_all=True, append_images=images[1:], duration=100, loop=0)

        # compress 3d data to 2d (using normalized data)
        w = np.stack([eigen.T[0], eigen.T[1]], axis = 1)
        data_pca = data @ w

        plt.figure()
        plt.title(f'{filename}_2D')
        plt.xlabel(f'PC1(c rate={c_rate[0]:.3f})')
        plt.ylabel(f'PC2(c rate={c_rate[1]:.3f})')
        plt.scatter(data_pca.T[0], data_pca.T[1], c = 'deeppink')
        # plt.savefig('fig/' + f'{filename}_2D.png')
        plt.show()
        plt.close()
        
    else:
        p_rate = np.zeros_like(c_rate)
        for i in range(c_rate.shape[0]):
            p_rate[i] = np.sum(c_rate[:i+1])  # cumulative contribution rate

        p_min = np.min(np.where(p_rate >= 0.9))
        
        plt.figure()
        plt.title('cumulative contribution rate')
        plt.xlabel('dimention')
        plt.ylabel('cumulative contribution rate')
        x = np.arange(1, p_rate.shape[0]+1)
        y = np.linspace(min(p_rate), max(p_rate), p_rate.shape[0])
        plt.plot(x, p_rate, color = 'k')
        plt.plot(x, np.full(p_rate.shape[0], p_rate[p_min]), color = 'c', linestyle = 'dashed', label = f'rate={p_rate[p_min]:.4f}')
        plt.plot(np.full(p_rate.shape[0], p_min), y, color = 'y', linestyle = 'dashed', label = f'dim={p_min+1}')
        plt.legend(loc='lower right')
        plt.tight_layout()
        # plt.savefig('fig/' + 'cumulative.png')
        plt.show()
        plt.close()
                

if __name__ == "__main__":
    main()