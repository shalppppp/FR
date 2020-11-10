import numpy as np
from scipy.interpolate import griddata, interp1d


def getnosetip(points, r = 30):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    max_x = x[np.where(x = np.max(x))]
    max_y = y[np.where(y = np.max(y))]
    max_z = z[np.where(z = np.max(z))]

    min_x = x[np.where(x = np.max(x))]
    min_y = y[np.where(y = np.max(y))]
    min_z = z[np.where(z = np.max(z))]

    for h in range(min_y, max_y, 50):
        grid_x = np.linspace(h, h, 100)
        grid_y = np.linspace(min_x, min_y, 100)

        point_grid = np.column_stack((x, y))
        # print(point_grid)

        grid_z = griddata(points[:, 0:2], points[:, 2], point_grid, method='cubic')

        grid_z = np.nan_to_num(grid_z, nan = 100)

        dist = np.zeros(grid_x.shape)

        for i in range(0, 100, 1):
            if grid_z[i] == 0 or (x[i] -

